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

You are not completing a prefix. You are *describing* the sentence you want,
and the system works out which sentence fits your keystrokes best.

| You type | You get |
| --- | --- |
| `clo` | `clothes` — an ordinary prefix |
| `clth` | `clothes` — letters left out |
| `brd` | `bread` — an abbreviation |
| `gt bck` | `get back to you as soon as possible` |
| `th wthr hs bn` | `The weather has been good this week` |
| `aprico` | `apricots` — a word the model would never have guessed |

Every suggestion carries its probability and its **match quality**, because
that is the signal you steer on: `exact` means it has you and you can stop
typing; `loose` means it is stretching, so add a letter — or delete one that
was a typo.

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

**The channel** is a Levenshtein grid whose costs are negative
log-probabilities of real typing errors: a substitution (cheaper between keys
that are physically adjacent on *your* layout), a deletion (a keystroke
nothing explains), and a skip (a character you did not type — this is what
makes abbreviation work). Characters past what you typed are free; that is
the prediction, not an error.

**The posterior drives the search, not just the ranking.** Extending a path
can only lower the prior and can only raise the channel's cost floor, so
`prior − cost_floor` is an upper bound on every string reachable below a
node. Best-first on that bound is A\* with an admissible heuristic: branches
that disagree with your keystrokes are abandoned after one token, branches
that agree are decoded many tokens deep. The unpromising strings are never
decoded at all.

### Three things that were not obvious

Each of these was a bug found by measurement, and each is documented at the
code that fixes it.

**Seed the token-aligned prefix, not the keystrokes.** Pruning cannot rescue
a string the model never proposed, so what you type is also inserted as a
starting path. Seeding the raw text does not work: `" apricot"` is spelled
`[" apr", "icot"]` but the partial `" aprico"` is `[" apr", "ico"]`, and from
`"ico"` the model's likeliest continuations are `"les"` and `"es"` — `"t"` is
not in its top eight. The natural route to the word never passes through the
partial word's token path. Seeding `" apr"` instead reaches it immediately.

**Re-price candidates under their canonical spelling.** A seeded branch
spells a partial word, which the model rates far below the natural spelling
of the finished one — measured at 8 nats for `" aprico"` versus `" apricot"`.
That is an artifact of an unnatural token split, not a statement about
apricots, so each finished candidate is re-priced under its own tokenization
and merged in.

**Search order must not be the sound bound.** Pure A\* on the bound collapses
to breadth-first, because extending a path only lowers its score — so a
one-token path like `"I"` always outranks the ten-token path that answers the
query. Against `"I will gt bck"` that is fatal: the channel makes a candidate
explain *every* keystroke, so short candidates are never emitted and the
search returns nothing while looking busy. Nodes are therefore ordered by
prior-spent-per-keystroke-covered. That estimate can be wrong, so it is
confined to the heap order — pruning and stopping keep using the admissible
bound, and a bad estimate can only cost time, never a candidate.

## Performance

Measured on a DGX Spark (GB10), `Qwen/Qwen3.5-0.8B-Base` in bf16:

| | |
| --- | --- |
| keystroke → updated ranking | **~5 ms** typical, ~16 ms worst case (pure Python, no GPU) |
| background decode of a new pool | ~1.7–3 s, 130–190 candidates |
| model load | ~20 s, once |

Re-ranking is `O(pool × query length × candidate length)`, so the worst case is
a long query against a full pool of sentence-length candidates — still inside a
single frame.

The two clocks are why the architecture looks the way it does. Every
keystroke re-ranks a **cached pool** — the prior is already known per
candidate, so a keystroke costs one Levenshtein grid each and nothing more.
The model is only consulted again when the pool stops explaining what you are
typing, which the channel cost detects directly.

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
| `--length-bonus` | nats per character; higher prefers longer sentences (`ctrl+s` cycles it live) |
| `--max-rounds` | batched forward passes per decode — the wall-clock lever |
| `--max-chars` | longest candidate to decode |

Press `f1` in the TUI for the keys.

## Tests

```bash
python -m pytest
```

99 tests, no GPU and no download: the search runs against a deterministic fake
model with a handful of string "tokens" and an explicit probability table,
which is what makes it possible to assert that three spellings of `"cat"` sum
to exactly 0.7 and that a pruned branch was never *explored* rather than
merely filtered.

## Licence

MIT.
