"""Transformers backend.

Three constraints shape this file.

**Vocabulary size.** Qwen3.5 has ~248k tokens, so one position of fp32 logits
is ~1 MB. Asking transformers for logits at every position of a batch would
cost gigabytes, so every forward passes ``logits_to_keep=1``.

**No padding.** ``logits_to_keep`` counts from the *end* of the tensor, which
is the wrong place for a right-padded short row. Sequences are bucketed by
length and each bucket stacked unpadded instead. Batches from the search share
a context and differ by a token or two, so the buckets stay large.

**Batch size is a latency knob, not a throughput knob.** Measured on a GB10:
batch 24 x 64 tokens takes ~85 ms, batch 48 takes ~384 ms. The superlinearity
is the 248k-wide output projection and its fp32 softmax, so going wider stops
paying. 24 is the default for that reason.

Qwen3.5 is a hybrid model -- most layers are linear-attention with recurrent
state rather than a forkable KV cache -- so the search re-forwards its shared
context each round instead of branching a cache. The context is short, and
this keeps the backend a single stateless call.
"""

from __future__ import annotations

import math
from bisect import bisect_left
from collections.abc import Sequence

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .lm import DEFAULT_MODEL, TopK

__all__ = ["DEFAULT_MODEL", "HFLanguageModel"]


def _buckets(sequences: Sequence[Sequence[int]]) -> dict[int, list[int]]:
    groups: dict[int, list[int]] = {}
    for i, seq in enumerate(sequences):
        groups.setdefault(len(seq), []).append(i)
    return groups


def _chunks(items: list[int], size: int):
    for start in range(0, len(items), size):
        yield items[start : start + size]


def _byte_decoder() -> dict[str, int]:
    """Inverse of GPT-2's byte-level BPE alphabet.

    Byte-level BPE maps every one of the 256 byte values to a printable
    character so the merge table can be plain text. Inverting that mapping
    recovers the exact bytes behind a token -- including the ones that are
    only half a UTF-8 character, which ``decode`` can only show as a
    replacement marker.
    """
    printable = (
        list(range(ord("!"), ord("~") + 1))
        + list(range(ord("\u00a1"), ord("\u00ac") + 1))
        + list(range(ord("\u00ae"), ord("\u00ff") + 1))
    )
    mapped = printable[:]
    n = 0
    for b in range(256):
        if b not in printable:
            printable.append(b)
            mapped.append(256 + n)
            n += 1
    return {chr(c): b for b, c in zip(printable, mapped)}


class HFLanguageModel:
    """A :class:`fuzzytype.lm.LanguageModel` backed by transformers."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL,
        *,
        device: str | None = None,
        dtype: "torch.dtype | None" = None,
        batch_size: int = 24,
    ) -> None:
        self.model_id = model_id
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        if dtype is None:
            dtype = torch.bfloat16 if self.device != "cpu" else torch.float32
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = (
            AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype)
            .to(self.device)
            .eval()
        )
        self.eos_token_id = self.tokenizer.eos_token_id
        self._token_bytes, self.special_token_ids = self._build_token_bytes()
        self._prefix_keys, self._prefix_ids = self._build_prefix_index()
        self._short_prefixes = self._build_short_prefixes()

    def _build_prefix_index(self) -> tuple[list[str], list[int]]:
        """Vocabulary sorted by its text, for "which words start with this".

        A whole word is very often a single token, and the model's ranking of
        that token can sit thousands of places down the distribution while
        still being a perfectly good guess: after this preamble "Hello" scores
        -14.2 against "Here" at -11.6, a gap of under three nats, yet it is
        nowhere near the top-64 the search expands. No amount of tree walking
        finds it, because it is one token away and simply never proposed.

        Looking it up in the vocabulary costs nothing at query time and one
        forward pass to price. Leading spaces and case are normalised away so
        that typing "hel" finds "Hello", " hello" and "Helsinki" alike.
        """
        pairs = sorted(
            (self._token_bytes[i].decode("utf-8", errors="replace").lstrip().lower(), i)
            for i in range(len(self._token_bytes))
            if i not in self.special_token_ids
        )
        return [k for k, _ in pairs], [i for _, i in pairs]

    def _build_short_prefixes(self, depth: int = 2, cap: int = 2048) -> dict:
        """Shortest-first token lists for one- and two-character prefixes.

        These are the ranges too large to order at query time -- and the ones
        that matter most, since a word abbreviated to a single letter is
        exactly the case that needs them.
        """
        buckets: dict[str, list[tuple[int, int]]] = {}
        for key, tid in zip(self._prefix_keys, self._prefix_ids):
            for n in range(1, min(depth, len(key)) + 1):
                buckets.setdefault(key[:n], []).append((len(key), tid))
        out = {}
        for prefix, entries in buckets.items():
            entries.sort()
            out[prefix] = tuple(tid for _, tid in entries[:cap])
        return out

    def tokens_with_prefix(self, prefix: str, limit: int = 512) -> list[int]:
        """Token ids whose text starts with ``prefix``, ignoring case and space.

        Ordered shortest-key first, because when only a few can be taken the
        short word is nearly always the one meant -- "how" before "however".

        Short prefixes are precomputed. Taking the first N of the alphabetical
        range and sorting *those* by length is a trap: thousands of tokens
        begin with "h", so the cut lands somewhere in "hab..." and "how" is
        never seen at all, which silently breaks abbreviation past the first
        word. The ordering has to be applied to the whole range.
        """
        key = prefix.lstrip().lower()
        if not key:
            return []
        bucket = self._short_prefixes.get(key)
        if bucket is not None:
            return list(bucket[:limit])
        found: list[tuple[int, int]] = []
        start = bisect_left(self._prefix_keys, key)
        for i in range(start, len(self._prefix_keys)):
            if not self._prefix_keys[i].startswith(key):
                break
            found.append((len(self._prefix_keys[i]), self._prefix_ids[i]))
        found.sort()
        return [i for _, i in found[:limit]]

    @torch.no_grad()
    def token_logprobs(
        self, sequence: Sequence[int], token_ids: Sequence[int]
    ) -> list[float]:
        """log P(token | sequence) for specific tokens, in one forward pass.

        The cost does not depend on how many tokens are asked about, which is
        what makes a wide vocabulary lookup affordable.
        """
        if not token_ids:
            return []
        ids = torch.tensor([list(sequence)], dtype=torch.long, device=self.device)
        logits = self.model(input_ids=ids, logits_to_keep=1).logits[:, -1, :]
        logprobs = torch.log_softmax(logits.float(), dim=-1)[0]
        wanted = torch.tensor(list(token_ids), dtype=torch.long, device=self.device)
        return logprobs.index_select(0, wanted).tolist()

    def _build_token_bytes(self) -> tuple[list[bytes], frozenset[int]]:
        """Per-token raw bytes, plus the ids the search must not decode through.

        Built once so the inner loop never calls the tokenizer: the search
        expands tens of thousands of children per keystroke-triggered decode,
        and a ``decode`` call each was measured as the single largest cost --
        larger than the forward passes.

        The result is verified against the tokenizer on a sample rather than
        trusted, because a wrong byte table would corrupt every candidate
        string silently instead of failing.
        """
        vocab_size = int(self.model.config.vocab_size)
        decoder = _byte_decoder()
        tokens = self.tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
        special = set(self.tokenizer.all_special_ids or ())
        special.update(self.tokenizer.get_added_vocab().values())
        table: list[bytes] = []
        for tid, tok in enumerate(tokens):
            if tok is None:  # unused id slot; never emit it
                table.append(b"")
                special.add(tid)
                continue
            try:
                table.append(bytes(decoder[c] for c in tok))
            except KeyError:
                # Not in the byte alphabet: an added token whose surface form
                # is literal text. Keep its bytes, and refuse to decode it.
                table.append(tok.encode("utf-8"))
                special.add(tid)

        checked = 0
        for tid in range(0, vocab_size, max(1, vocab_size // 512)):
            if tid in special:
                continue
            reference = self.tokenizer.decode([tid], skip_special_tokens=False)
            if "\ufffd" in reference:
                continue  # partial codepoint: decode cannot show the truth
            mine = table[tid].decode("utf-8", errors="replace")
            if mine != reference:
                raise RuntimeError(
                    "byte table disagrees with the tokenizer at id "
                    f"{tid}: {mine!r} != {reference!r}"
                )
            checked += 1
        if checked < 50:
            raise RuntimeError(
                f"byte table validated against only {checked} tokens; refusing "
                "to trust it"
            )
        return table, frozenset(special)

    def token_bytes(self, token_id: int) -> bytes:
        return self._token_bytes[token_id]

    def encode(self, text: str) -> list[int]:
        return self.tokenizer(text, add_special_tokens=False).input_ids

    def decode(self, token_ids: Sequence[int]) -> str:
        """Whole-sequence decode. The search uses :meth:`token_bytes` instead;
        this stays for callers that want a human-readable spelling."""
        return b"".join(self._token_bytes[i] for i in token_ids).decode(
            "utf-8", errors="replace"
        )

    @torch.no_grad()
    def top_next(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        top_k: int,
        top_p: float,
        extra_ids: "Sequence[Sequence[int]] | None" = None,
        extra_keep: int = 0,
    ) -> list[TopK]:
        """Truncated next-token distributions, plus any tokens asked for.

        ``extra_ids`` carries tokens the caller wants considered whatever
        their rank -- the words that match the keystrokes still unexplained.
        Their log-probabilities come out of the row already computed, so
        proposing many costs a gather rather than a forward pass, and only
        the best ``extra_keep`` are returned. That split matters: the caller
        cannot tell which of "h", "ha" or "hay" begins the next word, so it
        proposes all of them and lets the model say.
        """
        results: list[TopK | None] = [None] * len(sequences)
        for idxs in _buckets(sequences).values():
            for chunk in _chunks(idxs, self.batch_size):
                ids = torch.tensor(
                    [list(sequences[i]) for i in chunk],
                    dtype=torch.long,
                    device=self.device,
                )
                logits = self.model(input_ids=ids, logits_to_keep=1).logits[:, -1, :]
                logprobs = torch.log_softmax(logits.float(), dim=-1)
                k = min(top_k, logprobs.shape[-1])
                vals, inds = torch.topk(logprobs, k, dim=-1)
                probs = vals.exp()
                # topk is sorted descending, so "mass strictly before this
                # token < top_p" is a prefix mask and always keeps at least one.
                before = probs.cumsum(dim=-1) - probs
                kept = (before < top_p).sum(dim=-1).clamp(min=1)
                vals_l, inds_l, kept_l = vals.tolist(), inds.tolist(), kept.tolist()
                probs_l = probs.tolist()
                for row, i in enumerate(chunk):
                    n = int(kept_l[row])
                    ids = list(inds_l[row][:n])
                    lps = list(vals_l[row][:n])
                    wanted = list(extra_ids[i]) if extra_ids is not None else []
                    seen = set(ids)
                    wanted = [t for t in dict.fromkeys(wanted) if t not in seen]
                    mass = float(sum(probs_l[row][:n]))
                    if wanted and extra_keep > 0:
                        picked = torch.tensor(
                            wanted, dtype=torch.long, device=self.device
                        )
                        scored = logprobs[row].index_select(0, picked).tolist()
                        best = sorted(
                            zip(wanted, scored), key=lambda kv: -kv[1]
                        )[:extra_keep]
                        ids.extend(t for t, _ in best)
                        lps.extend(lp for _, lp in best)
                        # These are extra tokens *kept*, so they belong in the
                        # mass that says how much was kept. Their
                        # log-probabilities are untouched -- the row is a
                        # log_softmax over the whole vocabulary, so a token
                        # reached this way is worth exactly what it would have
                        # been worth inside the top-k.
                        mass += float(sum(math.exp(lp) for _, lp in best))
                    results[i] = TopK(
                        token_ids=tuple(ids),
                        logprobs=tuple(lps),
                        kept_mass=mass,
                    )
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:  # pragma: no cover - defensive
            raise RuntimeError(f"no distribution computed for rows {missing}")
        return results  # type: ignore[return-value]

    @torch.no_grad()
    def sequence_logprobs(
        self, items: Sequence[tuple[Sequence[int], Sequence[int]]]
    ) -> list[float]:
        """Price explicit continuations -- a seeded branch, or a whole pool.

        Continuations are padded to a common length within each batch rather
        than bucketed by length. Bucketing means one forward per distinct
        continuation length, each with whatever few rows happen to share it;
        padding means full batches and roughly half as many forwards. It is
        safe because ``logits_to_keep`` counts from the end and every padded
        row has the same total length, so the kept slice starts at the first
        continuation token for every row alike; the padding is then masked out
        of the sum rather than scored.
        """
        results = [float("nan")] * len(items)
        groups: dict[int, list[int]] = {}
        for i, (prefix, cont) in enumerate(items):
            if not prefix:
                raise ValueError("prefix must contain at least one token")
            if not cont:
                raise ValueError("continuation must contain at least one token")
            groups.setdefault(len(prefix), []).append(i)

        pad = self.eos_token_id or 0
        for plen, idxs in groups.items():
            for chunk in _chunks(idxs, self.batch_size):
                widest = max(len(items[i][1]) for i in chunk)
                rows = [
                    list(items[i][0])
                    + list(items[i][1])
                    + [pad] * (widest - len(items[i][1]))
                    for i in chunk
                ]
                full = torch.tensor(rows, dtype=torch.long, device=self.device)
                logits = self.model(input_ids=full, logits_to_keep=widest + 1).logits
                lp = torch.log_softmax(logits[:, :widest, :].float(), dim=-1)
                per_token = lp.gather(-1, full[:, plen:].unsqueeze(-1)).squeeze(-1)
                lengths = torch.tensor(
                    [len(items[i][1]) for i in chunk], device=self.device
                )
                keep = (
                    torch.arange(widest, device=self.device).unsqueeze(0)
                    < lengths.unsqueeze(1)
                )
                totals = (per_token * keep).sum(dim=1).tolist()
                for row, i in enumerate(chunk):
                    results[i] = float(totals[row])
        return results
