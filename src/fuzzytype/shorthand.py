"""Prompt mode: ask the model to expand the shorthand itself.

The channel in :mod:`fuzzytype.channel` is a hand-built model of how a typist
abbreviates -- what a skipped run costs, what a wrong key costs, how those
trade off. Every one of those numbers had to be calibrated, and each is a
guess about behaviour the language model has already seen far more of than we
have.

This mode asks the model instead. The keystrokes go into a prompt as an
example of shorthand, and the completion tree is walked after ``full text:``:

    shorthand: gt bck
    full text: get back to you as soon as possible

What comes back is ``P(expansion | shorthand, context)`` straight from the
model, so candidates are ranked by the model's own probabilities and nothing
needs calibrating. It also changes what the search *is*: there is no channel
to prune with, but the distribution is already conditioned on the shorthand,
so the likely continuations are the plausible expansions and ordinary
best-first enumeration finds them.

The two modes are not exclusive. The channel can still be applied on top as a
re-ranking term, which is worth having when the model's format-following is
shaky -- a 0.8B base model does sometimes answer a different question than
the one the prompt asks.

**Why the examples look like this.** Base models continue patterns, so the
prompt is a few worked pairs and then an unfinished one. They are drawn from
unrelated domains on purpose: examples close to what the user is writing pull
the answer towards their own content rather than teaching the format.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["Example", "DEFAULT_EXAMPLES", "build_prompt", "PROMPT_HEADER"]


@dataclass(frozen=True)
class Example:
    """One worked shorthand expansion."""

    context: str
    shorthand: str
    full: str


PROMPT_HEADER = (
    "Expanding typed shorthand into the full text the writer meant.\n"
    "The shorthand leaves letters out; the full text spells them back in.\n"
)

#: Deliberately varied and unrelated to anything a user is likely to type:
#: one word, one phrase, one whole clause, one proper noun.
DEFAULT_EXAMPLES: tuple[Example, ...] = (
    Example("The recipe calls for two ripe", "avcdo", "avocados"),
    Example("Please forward the invoice to", "acnts", "accounts"),
    Example("I am afraid I will have to", "cncl th mtg", "cancel the meeting"),
    Example("The train leaves from", "plfrm nne", "platform nine"),
)


def build_prompt(
    context: str,
    shorthand: str,
    examples: "tuple[Example, ...]" = DEFAULT_EXAMPLES,
    header: str = PROMPT_HEADER,
) -> str:
    """The text to continue. The walk starts immediately after it.

    Ends without a trailing space so the model chooses the leading space
    itself, exactly as it does in the worked examples.
    """
    blocks = [header]
    for example in examples:
        blocks.append(
            f"\ncontext: {example.context}\n"
            f"shorthand: {example.shorthand}\n"
            f"full text: {example.full}\n"
        )
    blocks.append(
        f"\ncontext: {context}\nshorthand: {shorthand}\nfull text:"
    )
    return "".join(blocks)
