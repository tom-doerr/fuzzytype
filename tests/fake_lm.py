"""A deterministic toy language model.

The search is the interesting part of fuzzytype and it should be testable
without a GPU, a download, or a 248k-token vocabulary. FakeLM is a handful of
string "tokens" and an explicit probability table keyed by the exact token
sequence so far -- which makes it possible to write a vocabulary where one
string has several spellings and then assert what the search does about it.
"""

from __future__ import annotations

import math

from fuzzytype.lm import TopK


class FakeLM:
    def __init__(self, vocab, table, context=(), eos_token="<eos>"):
        self.vocab = list(vocab)
        self.ids = {tok: i for i, tok in enumerate(self.vocab)}
        self.table = {k: dict(v) for k, v in table.items()}
        self.eos_token_id = self.ids[eos_token] if eos_token else None
        self.special_token_ids = frozenset(
            {self.eos_token_id} if eos_token else set()
        )
        self.document_start_id = self.eos_token_id or 0
        self.context = tuple(context)
        self._longest_first = sorted(self.vocab, key=len, reverse=True)
        #: every sequence the search asked to expand, so a test can assert a
        #: pruned branch was never *explored* rather than merely filtered out
        self.seen: list[tuple[str, ...]] = []

    # -- tokenizer -------------------------------------------------------
    def encode(self, text: str) -> list[int]:
        out, i = [], 0
        while i < len(text):
            for tok in self._longest_first:
                if tok and text.startswith(tok, i):
                    out.append(self.ids[tok])
                    i += len(tok)
                    break
            else:
                raise ValueError(f"cannot tokenize {text[i:]!r}")
        return out

    def decode(self, token_ids) -> str:
        return "".join(self.vocab[i] for i in token_ids)

    def token_bytes(self, token_id: int) -> bytes:
        return self.vocab[token_id].encode("utf-8")

    # -- model -----------------------------------------------------------
    def _key(self, token_ids):
        return tuple(self.vocab[i] for i in token_ids)

    def _state(self, sequence):
        if tuple(sequence[: len(self.context)]) != self.context:
            raise AssertionError("sequence does not start with the context")
        return self._key(sequence[len(self.context) :])

    def top_next(self, sequences, *, top_k, top_p, match_chars=None, match_top_k=0):
        results = []
        for index, seq in enumerate(sequences):
            state = self._state(seq)
            self.seen.append(state)
            dist = self.table.get(state)
            if dist is None:
                results.append(TopK(token_ids=(), logprobs=(), kept_mass=0.0))
                continue
            ranked = sorted(dist.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]
            kept, before = [], 0.0
            for tok, prob in ranked:
                if before >= top_p and kept:
                    break
                kept.append((tok, prob))
                before += prob
            chosen = {self.ids[t] for t, _ in kept}
            token_ids = [self.ids[t] for t, _ in kept]
            logprobs = [math.log(p) for _, p in kept]
            char = match_chars[index] if match_chars is not None else None
            if char and match_top_k > 0:
                starts = []
                for tok, prob in dist.items():
                    tid = self.ids[tok]
                    stripped = tok.lstrip().lower()
                    if tid not in chosen and stripped[:1] == char.lower():
                        starts.append((tid, math.log(prob)))
                starts.sort(key=lambda kv: -kv[1])
                for tid, lp in starts[:match_top_k]:
                    chosen.add(tid)
                    token_ids.append(tid)
                    logprobs.append(lp)
            results.append(
                TopK(
                    token_ids=tuple(token_ids),
                    logprobs=tuple(logprobs),
                    kept_mass=sum(math.exp(lp) for lp in logprobs),
                )
            )
        return results

    def sequence_logprobs(self, items):
        out = []
        for prefix, cont in items:
            state = list(self._state(prefix))
            total = 0.0
            for tid in cont:
                dist = self.table.get(tuple(state), {})
                prob = dist.get(self.vocab[tid], 0.0)
                if prob <= 0.0:
                    total = float("-inf")
                    break
                total += math.log(prob)
                state.append(self.vocab[tid])
            out.append(total)
        return out


#: A world where "cat" has three spellings, so merging can be asserted on an
#: exact number, and where "cart" shares a prefix with it.
#: The capitalised tokens exist because the engine offers a capitalised
#: spelling of whatever is typed as a seed, and a seed is really encoded.
CAT_VOCAB = [
    "<eos>", " ", ".", " cat", " ca", "t", " car", "s",
    # Bare and capitalised letters: the engine seeds the search with what was
    # typed, without a leading space at the start of a word and capitalised
    # as well, and a seed is really encoded.
    " C", "C", "c", "a", "r",
]
CAT_TABLE = {
    (): {" cat": 0.5, " ca": 0.2, " car": 0.3},
    (" cat",): {" ": 0.7, ".": 0.3},
    (" ca",): {"t": 1.0},
    (" ca", "t"): {" ": 1.0},
    (" car",): {" ": 0.6, "t": 0.4},
    (" car", "t"): {" ": 1.0},
    (" cat", " "): {"<eos>": 1.0},
    (" cat", "."): {"<eos>": 1.0},
    (" ca", "t", " "): {"<eos>": 1.0},
    (" car", " "): {"<eos>": 1.0},
    (" car", "t", " "): {"<eos>": 1.0},
}


def cat_lm(context=(1,)):
    """LM over CAT_TABLE. ``context`` is an arbitrary non-empty prefix."""
    return FakeLM(CAT_VOCAB, CAT_TABLE, context=context)


#: A world with one genuinely long word, so that channel pruning has
#: something long enough to prune. The error budget grows with the query but
#: the cost of an unmatched candidate grows with its *length*, so a four
#: letter vocabulary can never trigger the prune at all.
LONG_VOCAB = ["<eos>", " ", ".", " ele", "ph", "ant", " cat"]
LONG_TABLE = {
    (): {" ele": 0.5, " cat": 0.5},
    (" ele",): {"ph": 1.0},
    (" ele", "ph"): {"ant": 1.0},
    (" ele", "ph", "ant"): {" ": 1.0},
    (" ele", "ph", "ant", " "): {"<eos>": 1.0},
    (" cat",): {" ": 1.0},
    (" cat", " "): {"<eos>": 1.0},
}


def long_lm(context=(1,)):
    return FakeLM(LONG_VOCAB, LONG_TABLE, context=context)


#: A world with one genuinely long word, so that channel pruning has
#: something long enough to prune. The error budget grows with the query but
#: the cost of an unmatched candidate grows with its *length*, so a four
#: letter vocabulary can never trigger the prune at all.
LONG_VOCAB = ["<eos>", " ", ".", " ele", "ph", "ant", " cat"]
LONG_TABLE = {
    (): {" ele": 0.5, " cat": 0.5},
    (" ele",): {"ph": 1.0},
    (" ele", "ph"): {"ant": 1.0},
    (" ele", "ph", "ant"): {" ": 1.0},
    (" ele", "ph", "ant", " "): {"<eos>": 1.0},
    (" cat",): {" ": 1.0},
    (" cat", " "): {"<eos>": 1.0},
}


def long_lm(context=(1,)):
    return FakeLM(LONG_VOCAB, LONG_TABLE, context=context)


#: A world with one genuinely long word, so that channel pruning has
#: something long enough to prune. The error budget grows with the query but
#: the cost of an unmatched candidate grows with its *length*, so a four
#: letter vocabulary can never trigger the prune at all.
LONG_VOCAB = ["<eos>", " ", ".", " ele", "ph", "ant", " cat"]
LONG_TABLE = {
    (): {" ele": 0.5, " cat": 0.5},
    (" ele",): {"ph": 1.0},
    (" ele", "ph"): {"ant": 1.0},
    (" ele", "ph", "ant"): {" ": 1.0},
    (" ele", "ph", "ant", " "): {"<eos>": 1.0},
    (" cat",): {" ": 1.0},
    (" cat", " "): {"<eos>": 1.0},
}


def long_lm(context=(1,)):
    return FakeLM(LONG_VOCAB, LONG_TABLE, context=context)


#: A world with one genuinely long word, so that channel pruning has
#: something long enough to prune. The error budget grows with the query but
#: the cost of an unmatched candidate grows with its *length*, so a four
#: letter vocabulary can never trigger the prune at all.
LONG_VOCAB = ["<eos>", " ", ".", " ele", "ph", "ant", " cat"]
LONG_TABLE = {
    (): {" ele": 0.5, " cat": 0.5},
    (" ele",): {"ph": 1.0},
    (" ele", "ph"): {"ant": 1.0},
    (" ele", "ph", "ant"): {" ": 1.0},
    (" ele", "ph", "ant", " "): {"<eos>": 1.0},
    (" cat",): {" ": 1.0},
    (" cat", " "): {"<eos>": 1.0},
}


def long_lm(context=(1,)):
    return FakeLM(LONG_VOCAB, LONG_TABLE, context=context)


def _prefix_methods(cls):
    """Vocabulary lookup, so the fake supports prefix seeding too."""

    def tokens_with_prefix(self, prefix, limit=512):
        key = prefix.lstrip().lower()
        if not key:
            return []
        out = [
            i
            for i, tok in enumerate(self.vocab)
            if i not in self.special_token_ids
            and tok.lstrip().lower().startswith(key)
        ]
        return out[:limit]

    def token_logprobs(self, sequence, token_ids):
        import math as _math

        dist = self.table.get(self._state(sequence), {})
        return [
            _math.log(dist[self.vocab[i]]) if dist.get(self.vocab[i], 0.0) > 0
            else float("-inf")
            for i in token_ids
        ]

    cls.tokens_with_prefix = tokens_with_prefix
    cls.token_logprobs = token_logprobs
    return cls


#: A world where one whole word is a single token that the walk cannot reach,
#: because it sits outside the top-k the search expands. Only a vocabulary
#: lookup finds it -- which is the "hel" -> "Hello" case.
PREFIX_VOCAB = ["<eos>", " ", ".", " cat", " car", " cart", "s"]
PREFIX_TABLE = {
    (): {" cat": 0.6, " car": 0.3, " cart": 0.05},
    (" cat",): {" ": 1.0},
    (" car",): {" ": 1.0},
    (" cart",): {" ": 1.0},
    (" cat", " "): {"<eos>": 1.0},
    (" car", " "): {"<eos>": 1.0},
    (" cart", " "): {"<eos>": 1.0},
}


def prefix_lm(context=(1,)):
    return FakeLM(PREFIX_VOCAB, PREFIX_TABLE, context=context)
