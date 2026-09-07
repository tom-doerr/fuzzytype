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
        self.document_start_id = (
            self.tokenizer.bos_token_id
            if self.tokenizer.bos_token_id is not None
            else self.tokenizer.eos_token_id
        )
        self._token_bytes, self.special_token_ids = self._build_token_bytes()
        self._first_char = self._build_first_char()

    def _build_first_char(self) -> "torch.Tensor":
        """The first letter each token would write, as a code point per id.

        Lets the search take a second top-k over *only* the tokens that could
        begin the next word, which is an exact top-k over the whole
        vocabulary rather than a lookup into part of it.
        """
        codes = []
        for raw in self._token_bytes:
            text = raw.decode("utf-8", errors="replace").lstrip().lower()
            codes.append(ord(text[0]) if text and text[0].isalnum() else -1)
        return torch.tensor(codes, dtype=torch.int32, device=self.device)

    @torch.no_grad()
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
        match_chars: "Sequence[str | None] | None" = None,
        match_top_k: int = 0,
    ) -> list[TopK]:
        """Truncated next-token distributions, one per input sequence.

        A plain top-k selects on the prior alone, and that is the wrong
        question when something has been typed: pruning cannot rescue a token
        that was never proposed, and the token wanted is often nowhere near
        the top. Measured, "Hello" sits under three nats from "Here" and far
        outside the top sixty-four, so typing "hel" returned every "Here ..."
        and no "Hello" however long the search ran.

        So when ``match_chars`` names the letter the next word must begin
        with, a second top-k is taken over *only* the tokens that begin with
        it. Both are exact top-k over the whole 248k-token distribution the
        forward pass already produced -- the first asks "what would the model
        write", the second "what would it write that starts the way you
        typed" -- and the union is returned. No threshold and no weighting is
        involved, so nothing is repriced.
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
                matched = self._matching_top_k(logprobs, chunk, match_chars, match_top_k)
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
                    seen = set(ids)
                    mass = float(sum(probs_l[row][:n]))
                    for token, logprob in matched[row]:
                        if token in seen:
                            continue
                        seen.add(token)
                        ids.append(token)
                        lps.append(logprob)
                        # Kept, so it counts towards how much was kept. Both
                        # selections come from the same log_softmax over the
                        # whole vocabulary, so a token reached this way is
                        # worth exactly what it was worth in the first.
                        mass += math.exp(logprob)
                    results[i] = TopK(
                        token_ids=tuple(ids),
                        logprobs=tuple(lps),
                        kept_mass=mass,
                    )
        missing = [i for i, r in enumerate(results) if r is None]
        if missing:  # pragma: no cover - defensive
            raise RuntimeError(f"no distribution computed for rows {missing}")
        return results  # type: ignore[return-value]

    def _matching_top_k(
        self,
        logprobs: "torch.Tensor",
        chunk: list[int],
        match_chars: "Sequence[str | None] | None",
        match_top_k: int,
    ) -> list[list[tuple[int, float]]]:
        """The likeliest tokens that *begin* with each row's wanted letter."""
        blank: list[list[tuple[int, float]]] = [[] for _ in chunk]
        if match_chars is None or match_top_k < 1:
            return blank
        wanted = [match_chars[i] for i in chunk]
        if not any(wanted):
            return blank
        codes = torch.tensor(
            [ord(c.lower()) if c else -2 for c in wanted],
            dtype=torch.int32,
            device=self.device,
        ).unsqueeze(1)
        allowed = self._first_char.unsqueeze(0) == codes
        masked = logprobs.masked_fill(~allowed, float("-inf"))
        k = min(match_top_k, masked.shape[-1])
        vals, inds = torch.topk(masked, k, dim=-1)
        vals_l, inds_l = vals.tolist(), inds.tolist()
        return [
            [
                (token, logprob)
                for token, logprob in zip(inds_l[row], vals_l[row])
                if logprob != float("-inf")
            ]
            for row in range(len(chunk))
        ]

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
