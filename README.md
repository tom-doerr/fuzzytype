# fuzzytype

Type loosely. Pick the sentence you meant.

A base language model decodes the likely continuations of what you have
written. Your keystrokes are read as a **noisy transmission** of one of them.
The two combine into `P(you meant this)` — which both ranks the suggestions
and decides which branches are worth decoding further.

```
$ fuzzytype
┌──────────────────────────────────────────────────────────────────┐
│ Thanks for the update. I will gt bck█                             │
├──────────────────────────────────────────────────────────────────┤
│  #  P(meant)  match     suggestion                               │
│  1    58.2%   1 slip    get back to you as soon as possible      │
│  2    20.9%   1 slip    get back to you as soon as I can         │
│  3     6.1%   1 slip    get back to you with more information    │
└──────────────────────────────────────────────────────────────────┘
```

## The idea

You are not completing a prefix and you are not correcting typos. You are
*spending fewer keystrokes*: describe the sentence you want, and the system
works out which one fits what you typed. Omitting characters is the intended
way to use it, so it is the cheapest thing the model charges for; typo
tolerance is a fallback, and costs more.

| You type | You get |
| --- | --- |
| `clo` | `clothes` — an ordinary prefix |
| `clth` | `clothes` — letters left out |
| `brd` | `bread` — an abbreviation |
| `gt bck` | `get back to you as soon as possible` |
| `th wthr hs bn` | `The weather has been good this week` |
| `aprico` | `apricots` — a word the model would never have guessed |
| `hel` | `Hello`, `Hello everyone`, `Help` — the `l` is treated as evidence |
| `th wthr hs bn` | `The weather has been good this week` |

Every suggestion carries its probability and its **match quality**, because
that is the signal you steer on: `exact` means it has you and you can stop
typing; `loose` means it is stretching, so add a letter — or delete one that
was a typo.

## Two modes

**`--mode channel`** (default) ranks with a hand-built model of how a typist
abbreviates — what a skipped run costs, what a wrong key costs.

**`--mode prompt`** asks the language model to do the expanding instead. The
keystrokes go into a prompt as an example of shorthand and the completion
tree is walked after `full text:`, so candidates carry
`P(expansion | shorthand, context)` straight from the model and nothing needs
calibrating.

Measured on the same cases, 8 rounds each:

| typed | channel | prompt |
| --- | --- | --- |
| `brd` | bread @1, 36% | bread @1, **80%** |
| `th wthr hs bn` | @1, 14% | @1, **41%** |
| `gt bck` | @1, 26% | @1, 30% |
| `hel` | Hello @1 | help @1, hello @2 |
| `helhay` | Hello everyone | hello hay |

Prompt mode is markedly more confident wherever the input is genuinely
shorthand, and it finds more candidates per decode — at roughly 1.5× the time,
because the prompt makes every forward pass longer. It is also less literal
about single fragments: it reads `hel` as shorthand for `help` before `hello`,
which is defensible but not always what you meant.

**The channel still runs underneath it**, at reduced weight. The prompt on its
own never insists that a candidate account for *all* the keystrokes, so
`thiisatest` came back as `this test` at 57% with nothing to say the rest had
been ignored — and with no channel cost there is no match quality to show and
nothing to highlight. But at full strength the shorthand gets counted twice
and literal echoes win: `thisisatest` and `theisatest` climb to 12% and 8%.
Measured across that range, **0.6** keeps `this is test` and `this is a test`
on top without either failure. Tune with `--channel-weight`, or switch it off
with `--no-channel-assist`.

## How it works

Two log-probabilities, in nats, added:

```
score(c) = log P(c | context)   +   log P(keystrokes | c)
           └─ base LM prior ─┘       └─ noisy channel ─┘
```

**The prior** comes from a best-first walk of the model's completion tree.
One string can be spelled by several token sequences, and those are merged
with `logsumexp`, so a two-token and a one-token spelling of the same word
compete as one candidate on the total probability of the string.

**The channel** is an alignment grid whose costs are negative
log-probabilities of real typing behaviour: a substitution (cheaper between
keys physically adjacent on *your* layout), a deletion (a keystroke nothing
explains — the dearest, because you pressed that key on purpose), and a
**gap**, the characters you did not type.

Gaps are charged per *run*, not per character, and that is the difference
between the tool working and not. A flat per-character rate says every
omitted letter is an independent accident, which is not how anyone
abbreviates. Typing `helhay` for "hello how are you" drops eleven characters
in four runs — `hel[lo ]h[ow ]a[re ]y[ou]` — which at a flat rate came to 25.3
nats and lost to explaining the same keystrokes as three unrelated
*substitutions* at 12.0. The real reading was literally more expensive than
nonsense. With an affine gap (2.0 to open, 0.35 to continue) it costs 8.1 and
wins. Characters past what you typed stay free; that is the prediction, not
an error.

**The posterior drives the search, not just the ranking.** Extending a path
can only lower the prior and can only raise the channel's cost floor, so
`prior − cost_floor` is an upper bound on every string reachable below a
node. Best-first on that bound is A\* with an admissible heuristic: branches
that disagree with your keystrokes are abandoned after one token, branches
that agree are decoded many tokens deep. The unpromising strings are never
decoded at all.

### Six things that were not obvious

Each of these was a bug found by measurement, and each is documented at the
code that fixes it.

**Seed the token-aligned prefix, not the keystrokes.** Pruning cannot rescue
a string the model never proposed, so what you type is also inserted as a
starting path. Seeding the raw text does not work: `" apricot"` is spelled
`[" apr", "icot"]` but the partial `" aprico"` is `[" apr", "ico"]`, and from
`"ico"` the model's likeliest continuations are `"les"` and `"es"` — `"t"` is
not in its top eight. The natural route to the word never passes through the
partial word's token path. Seeding `" apr"` instead reaches it immediately.

**A length preference must never become the ranking.** An honest posterior
always prefers the shortest completion, so length has to be credited back or
every suggestion is one word long. Credit *linear* in length is unbounded, and
past some point it simply decides the outcome: against `hel`, a 27-character
`"Here is what I have written"` collected 10.8 nats, more than the cost of
ignoring the `l` entirely. Capping it flat does not work either — a cap low
enough to protect the match is also low enough that every sentence hits it and
length stops ordering anything. The credit is logarithmic instead, so going
from five characters to twenty-five is worth a lot and from forty to sixty
very little, which is also how useful the extra text actually is.

**Re-price candidates under their canonical spelling.** A seeded branch
spells a partial word, which the model rates far below the natural spelling
of the finished one — measured at 8 nats for `" aprico"` versus `" apricot"`.
That is an artifact of an unnatural token split, not a statement about
apricots, so each finished candidate is re-priced under its own tokenization
and merged in.

**Never hand the model a context that ends in a space.** Spaces belong to the
*following* token — `" bread"` is one token — so a context that already holds
the space forces the rarer spaceless spelling and lands the model somewhere it
has hardly been. Measured after `"...to buy some "`, its likeliest
continuations are the digits `1`, `2`, `3` and word fragments; drop the space
and they are `" new"`, `" more"`, `" clothes"`. The space is kept out of the
context and supplied by the candidate, which carries its own. This one is easy
to never notice: it costs quality everywhere rather than failing anywhere, and
it showed up as a stray `1` sitting at 16% among the idle suggestions. Fixing
it moved `par` → `Paris` from 49% to 91%.

**A prefix lookup must order the whole range, not the first slice of it.**
Thousands of tokens begin with `h`, so taking the first 512 alphabetically and
*then* preferring short ones lands the cut somewhere in `hab…` — `how` is
never even proposed, and abbreviation silently dies after the first word.
The ordering has to be applied to the entire range, which is why short
prefixes are precomputed.

**A whole word is often one token the search will never propose.** Pruning
and seeding both operate on the tree, and neither can reach a word the model
does not offer. After the default preamble `"Hello"` scores −14.2 against
`"Here"` at −11.6 — under three nats apart, an entirely reasonable guess — yet
nowhere near the top-64 the search expands. So typing `hel` returned every
`"Here ..."` and no `"Hello"`, and adding the `l` made it *worse*, because the
letter could then only be charged as a slip. The vocabulary already knows the
word: candidates are found by prefix, priced by a single forward pass whose
cost does not depend on how many were found, and seeded into the search to
compete on the same posterior as everything else.

**Search order must not be the sound bound.** Pure A\* on the bound collapses
to breadth-first, because extending a path only lowers its score — so a
one-token path like `"I"` always outranks the ten-token path that answers the
query. Against `"I will gt bck"` that is fatal: the channel makes a candidate
explain *every* keystroke, so short candidates are never emitted and the
search returns nothing while looking busy. Nodes are therefore ordered by
prior-spent-per-keystroke-covered. That estimate can be wrong, so it is
confined to the heap order — pruning and stopping keep using the admissible
bound, and a bad estimate can only cost time, never a candidate.

## What it does not do yet

Dense multi-word abbreviation with no separators does not reliably resolve.
Typing `helhay` for "hello how are you" returns "Hello everyone": the channel
scores the right sentence best (8.1 against 9.0), the vocabulary proposes
`How` at rank **1 of 957** at the right word boundary, and the node
`" Hello, how"` is genuinely built — it is simply never *expanded*, because
the frontier holds tens of thousands of nodes and only ~960 expansions
happen. Narrowing the search to 4,000 nodes does not fix it either, so the
cause is the priority ordering rather than the width: best-first on a joint
prior structurally prefers breadth, and four words deep is a long way down.

Separating the fragments (`gt bck`, `th wthr hs bn`) works well, because each
space anchors a word. The fix for the dense case is a coverage-stratified
beam — keeping a beam per number of keystrokes explained, rather than one
global priority queue.

## Performance

Measured on a DGX Spark (GB10), `Qwen/Qwen3.5-0.8B-Base` in bf16:

| | |
| --- | --- |
| keystroke → updated ranking | **~5 ms** typical, ~16 ms worst case (pure Python, no GPU) |
| background decode of a new pool | ~2–15 s, 150–500 candidates, published as it goes |
| model load | ~20 s, once |

Re-ranking is `O(pool × query length × candidate length)`, so the worst case is
a long query against a full pool of sentence-length candidates — still inside a
single frame.

Decoding runs long on purpose — more rounds means more and longer phrases —
so results are **published as they are found** rather than at the end, and the
status line shows the search working. It is also interruptible: a keystroke
the current pool cannot explain abandons a decode that is already out of date
rather than queueing behind it.

The two clocks are why the architecture looks the way it does. Every
keystroke re-ranks a **cached pool** — the prior is already known per
candidate, so a keystroke costs one alignment grid each and nothing more.
The model is only consulted again when the pool stops explaining what you are
typing, which the channel cost detects directly.

**The pool accumulates.** In channel mode a candidate's prior is
`P(text | context)` and does not depend on the keystrokes at all, so
everything found under an earlier query is still exactly as probable. Editing
the query used to throw all of it away and re-derive the same phrases from
nothing — which is what put a ceiling on how long a sentence could be built
up. Typing `gt bck` one character at a time: **267 candidates in 16.8 s
accumulating, against 118 in 20.3 s discarding**.

**Prompt mode re-prices instead of re-searching.** Its priors *do* depend on
the keystrokes, so a new character invalidates the numbers but not the
strings. Pricing the known candidates against the new prompt costs **2.7×
less than walking for them again** (10.1 s vs 27.3 s for 465 candidates), and
it is bounded by `max_rescore` so a keystroke does not wait on a long tail
nobody will read. Batches are padded to a common continuation length rather
than bucketed by length, which roughly halves the number of forward passes.

The batch size is a latency knob, not a throughput one: batch 24 × 64 tokens
takes ~85 ms, batch 48 takes ~384 ms. The superlinearity is the 248k-wide
output projection and its fp32 softmax, so going wider stops paying.

The single largest cost was once the tokenizer, not the GPU — the search
builds tens of thousands of children per decode, and a `decode` call each cost
more than the forward passes. Children are now built by concatenating raw
token **bytes**, which also handles tokens that end mid-UTF-8-character.

## Install

```bash
pipx install --system-site-packages --editable ~/git/fuzzytype
```

`--system-site-packages` is load-bearing: a sealed virtualenv would fetch a
`torch` wheel that does not match this machine's CUDA.

## Use

```bash
fuzzytype                      # the interactive TUI
fuzzytype predict --context "I went to the shop to buy some" --typed "brd"
fuzzytype bench                # decode and per-keystroke timings
```

Useful options:

| Option | Meaning |
| --- | --- |
| `--layout colemak-dh\|qwerty\|none` | which keys count as neighbours when scoring a typo |
| `--length-bonus` | how much longer candidates are preferred; saturating, 0 disables (`ctrl+s` cycles it live) |
| `--mode channel\|prompt` | rank with the typing model, or let the LM expand the shorthand |
| `--max-rounds` | batched forward passes per decode — more rounds finds more and longer phrases (`ctrl+up` / `ctrl+down` adjusts it live) |
| `--max-chars` | longest candidate to decode |

### Editing

`left` / `right` move the cursor through what you are typing, and keep going
into text you have already accepted. The suggestions follow it: go back to fix
an earlier word and you get suggestions for *that* point in the sentence
rather than the end.

```
I went to the shop to buy some ▏bread and
└──── behind the cursor ──────┘ └─ ahead ─┘
```

Everything behind the cursor is what the model reads and continues from.
Everything ahead is kept exactly as it is and re-attached afterwards, but it
does not affect the suggestions — the model can only read forwards.

`home` / `end` jump to either side; `backspace` and `delete` cut behind and
ahead.

Press `f1` in the TUI for the keys.

## Tests

```bash
python -m pytest
```

142 tests, no GPU and no download: the search runs against a deterministic fake
model with a handful of string "tokens" and an explicit probability table,
which is what makes it possible to assert that three spellings of `"cat"` sum
to exactly 0.7 and that a pruned branch was never *explored* rather than
merely filtered.

## Licence

MIT.
