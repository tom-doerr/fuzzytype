"""The language-model surface the search talks to.

Narrow on purpose: the search needs one thing from a model, "what are the
likely next tokens after this sequence". Keeping it a Protocol lets the whole
posterior search be tested against a deterministic fake with no GPU.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

# Lives here rather than in hf.py so argument parsing can name the default
# model without importing torch. `fuzzytype --help` should not build a CUDA
# context.
DEFAULT_MODEL = "Qwen/Qwen3.5-0.8B-Base"


@dataclass(frozen=True)
class TopK:
    """Truncated next-token distribution at one position.

    ``kept_mass`` is the probability of the tokens actually returned, reported
    rather than discarded so callers can distinguish "the model was confident"
    from "we threw away half the distribution to keep the search small".
    """

    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...]
    kept_mass: float

    def __post_init__(self) -> None:
        if len(self.token_ids) != len(self.logprobs):
            raise ValueError("token_ids and logprobs must have equal length")


class LanguageModel(Protocol):
    """Minimal causal-LM surface used by the predictor."""

    eos_token_id: int | None
    #: Tokens that are markup rather than text (chat markers, <think>, ...).
    #: The search refuses to decode through them.
    special_token_ids: frozenset[int]

    def encode(self, text: str) -> list[int]:
        """Tokenize ``text`` without adding special tokens."""
        ...

    def decode(self, token_ids: Sequence[int]) -> str:
        """Decode a full token sequence.

        Must be safe on partial sequences: byte-level BPE can split a UTF-8
        codepoint across tokens, so callers decode the whole sequence rather
        than concatenating per-token strings.
        """
        ...

    def token_bytes(self, token_id: int) -> bytes:
        """The raw bytes one token contributes.

        The search concatenates these instead of calling the tokenizer per
        node. That is not just an optimisation: a byte-level BPE token can end
        mid-UTF-8-codepoint, and accumulating bytes lets the next token
        complete the character instead of stranding a replacement marker.
        """
        ...

    def top_next(
        self,
        sequences: Sequence[Sequence[int]],
        *,
        top_k: int,
        top_p: float,
        extra_ids: Sequence[Sequence[int]] | None = None,
        extra_keep: int = 0,
    ) -> list[TopK]:
        """Truncated next-token distributions, one per input sequence.

        ``extra_ids`` proposes tokens to include whatever their rank -- the
        words matching the keystrokes still unexplained -- of which the best
        ``extra_keep`` are returned. Proposing is cheap and the model does
        the choosing.
        """
        ...

    def tokens_with_prefix(self, prefix: str, limit: int = 512) -> list[int]:
        """Token ids whose text starts with ``prefix``, ignoring case and space.

        A whole word is often a single token that the model ranks far below
        the search's cut-off while still being a good guess, so the search
        cannot reach it by walking the tree. Asking the vocabulary directly
        is what makes typing "hel" able to reach "Hello".
        """
        ...

    def token_logprobs(
        self, sequence: Sequence[int], token_ids: Sequence[int]
    ) -> list[float]:
        """log P(token | sequence) for specific tokens, in one forward pass."""
        ...

    def sequence_logprobs(
        self, items: Sequence[tuple[Sequence[int], Sequence[int]]]
    ) -> list[float]:
        """Total log P(continuation | prefix) for each ``(prefix, continuation)``.

        Used to price a seeded branch honestly: the search may insert the
        literal keystrokes as a path, and that path needs its true prior
        rather than an assumed one.
        """
        ...
