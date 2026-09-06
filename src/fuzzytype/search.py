"""Posterior-guided walk of the completion tree.

The question this answers is not "what would the model write next" but "given
what has been typed so far, which strings did the typist most likely mean".
That is a posterior, built from two log-probabilities that are simply added:

    score(c) = log P(c | context)  +  log P(keystrokes | c)
               \\_ base LM prior _/    \\_ noisy channel (channel.py) _/

**Why the posterior also drives the search.** Extending a path can only lower
the LM prior (probabilities multiply) and can only raise the channel's cost
floor (costs are non-negative). So for any node,

    bound = node_logprob - cost_bound(node)

is an upper bound on the score of *every* string reachable below it. A
best-first frontier ordered by that bound is A\\* with an admissible
heuristic: the token budget is spent where the posterior actually is. The
tree that results is ragged in exactly the way it should be -- a branch that
disagrees with the keystrokes is abandoned after one token, while a branch
that agrees is decoded many tokens deep. Nothing is filtered after the fact;
the unpromising strings are never decoded at all.

The same bound gives a stopping proof: once the frontier's best bound falls
below the k-th best finished score, no unexplored path can enter the top k.

**Seeding.** Pruning cannot rescue a string the model never proposed. If you
type a rare word, its first token may sit outside every node's top-k and the
search will find nothing at all. So the literal keystrokes are also inserted
as a starting path, priced with a real forward pass so it competes on an
honest prior rather than an assumed one. That is what guarantees a word you
are actually typing can always be completed.

**Where candidates come from.** A path becomes a candidate when its text ends
in a terminator, because that is the first moment the model has committed to
a word being over -- "quick" and "quickly" are indistinguishable until the
next character arrives. One string can be spelled by several token sequences;
those are merged with logsumexp so a two-token and a one-token spelling
compete as one candidate on the total probability of the string.
"""

from __future__ import annotations

import heapq
import math
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from .channel import (
    ChannelCosts,
    Grid,
    grid_values,
    initial_column,
    partial_cost,
    push_candidate_char,
)
from .lm import LanguageModel

__all__ = ["Candidate", "PredictConfig", "PredictStats", "predict"]

NEG_INF = float("-inf")

#: A path is only offered as a candidate once one of these has been emitted,
#: which is the evidence that the model considers the word finished.
TERMINATORS = frozenset(" \t\n\r.,;:!?)]}")
_TERMINATOR_STR = "".join(sorted(TERMINATORS))

#: A newline ends the field being typed; nothing past it is a suggestion.
_HARD_STOP = ("\n", "\r")

#: What str() shows for bytes that are not yet a whole UTF-8 character.
_PARTIAL = "�"


def logsumexp(values: Sequence[float]) -> float:
    """Numerically stable log(sum(exp(v))). Empty input -> -inf."""
    vals = [v for v in values if v != NEG_INF]
    if not vals:
        return NEG_INF
    top = max(vals)
    return top + math.log(sum(math.exp(v - top) for v in vals))


@dataclass(frozen=True)
class Candidate:
    """One string the typist may have meant.

    ``raw`` is what gets inserted, leading space and all; ``text`` is the
    trimmed form that is matched and displayed. ``consumed`` is how many
    characters of ``text`` the keystrokes account for -- the rest is
    prediction, which is what the UI highlights.
    """

    text: str
    raw: str
    logprob: float  # log P(text | context), merged over spellings
    cost: float  # -log P(keystrokes | text)
    consumed: int
    #: Keystrokes this candidate accounts for. Accepting it consumes exactly
    #: these, leaving the rest for whatever is suggested next.
    keystrokes: int
    n_paths: int
    tokens: tuple[int, ...]

    @property
    def score(self) -> float:
        """Unnormalised log posterior: prior plus likelihood."""
        return self.logprob - self.cost

    @property
    def prior(self) -> float:
        return math.exp(self.logprob)


@dataclass
class PredictConfig:
    """Search knobs. Defaults are tuned for a 0.8B base model on one GPU."""

    k: int = 12
    #: Hard ceiling on a candidate's length in characters. Long enough for a
    #: clause, since the list is meant to offer whole sentences.
    max_chars: int = 64
    max_tokens: int = 16
    #: Next tokens considered per node. Generous because the channel prunes
    #: the mismatches immediately and children are cheap to build.
    child_top_k: int = 64
    child_top_p: float = 0.9995
    batch_size: int = 24
    #: How long to keep looking. Each round is one batched forward pass, and
    #: more rounds means more and longer phrases. It can afford to be generous
    #: because results are published as they are found and the search can be
    #: interrupted the moment they stop being wanted.
    max_rounds: int = 40
    #: Publish the ranking so far every this many rounds.
    publish_every: int = 3
    max_expansions: int = 4000
    #: Absolute backstop only. A *prior* floor fights the whole design -- a
    #: rare word has a low prior and a perfect channel match, which is exactly
    #: the case worth decoding -- so the real pruning is the adaptive cutoff
    #: below, and this just stops hopeless paths consuming memory.
    min_logprob: float = -32.0
    #: Insert the literal keystrokes as a starting path, so a word the model
    #: would never have proposed is still reachable.
    seed_query: bool = True
    #: How many vocabulary entries beginning with the keystrokes to consider,
    #: and how many of them to seed once priced. Pricing is one forward pass
    #: regardless of the first number, so it can afford to be generous.
    seed_vocab_limit: int = 512
    seed_vocab_top: int = 24
    #: At every word boundary, how many vocabulary entries matching the
    #: *still unexplained* keystrokes to propose, and how many of those the
    #: model's own ranking keeps. This is what makes abbreviation work past
    #: the first word.
    boundary_vocab_limit: int = 512
    boundary_vocab_top: int = 16
    #: Re-price this many finished candidates under their canonical
    #: tokenization. 0 disables it.
    rescore_top: int = 24
    #: Nats charged per keystroke a path has yet to explain, so long
    #: candidates can compete with short ones for the frontier. Affects search
    #: order only -- never pruning. See :meth:`_Node.priority`.
    #: 2.0 by measurement: on "th wthr hs bn" a penalty of 0.6 finds nothing
    #: in 3.8s while 2.0 finds "The weather has been good this week" in 2.0s,
    #: and no other case regresses. It cannot affect idle prediction at all,
    #: because with nothing typed no keystrokes are owed.
    progress_penalty: float = 2.0
    #: Keep searching until an unexplored path could contribute at most this
    #: fraction of the k-th best score. 1.0 disables the early stop.
    stop_margin: float = 0.05

    def __post_init__(self) -> None:
        if not 0.0 < self.stop_margin <= 1.0:
            raise ValueError("stop_margin must be in (0, 1]")
        if self.k < 1:
            raise ValueError("k must be at least 1")


@dataclass
class PredictStats:
    """What the search did -- including everything it refused to decode."""

    rounds: int = 0
    expansions: int = 0
    nodes_pushed: int = 0
    pruned_by_channel: int = 0
    pruned_by_prior: int = 0
    pruned_by_cutoff: int = 0
    truncated: int = 0
    duplicates: int = 0
    seeded: int = 0
    rescored: int = 0
    seconds_model: float = 0.0
    seconds_total: float = 0.0
    distinct: int = 0
    frontier_bound: float = NEG_INF
    exhausted: bool = False
    complete_top_k: bool = False
    #: Abandoned early because the caller asked, normally because a keystroke
    #: made this decode out of date.
    interrupted: bool = False

    def as_dict(self) -> dict[str, object]:
        return {
            "rounds": self.rounds,
            "expansions": self.expansions,
            "nodes_pushed": self.nodes_pushed,
            "pruned_by_channel": self.pruned_by_channel,
            "pruned_by_prior": self.pruned_by_prior,
            "pruned_by_cutoff": self.pruned_by_cutoff,
            "truncated": self.truncated,
            "duplicates": self.duplicates,
            "seeded": self.seeded,
            "rescored": self.rescored,
            "seconds_model": round(self.seconds_model, 3),
            "seconds_total": round(self.seconds_total, 3),
            "distinct": self.distinct,
            "frontier_bound": self.frontier_bound,
            "exhausted": self.exhausted,
            "complete_top_k": self.complete_top_k,
            "interrupted": self.interrupted,
        }


@dataclass
class _Node:
    """A partial continuation, with its channel state carried alongside.

    ``data`` is raw bytes rather than text because a byte-level BPE token can
    end mid-character; accumulating bytes lets the next token finish it. The
    channel column is maintained incrementally, so keeping the fuzzy match in
    step with the decode costs O(len(query)) per new character.
    """

    tokens: tuple[int, ...]
    logprob: float
    data: bytes
    text: str
    column: Grid
    best_cost: float  # best match achieved at any prefix of this text
    best_consumed: int

    @property
    def query_len(self) -> int:
        return len(self.column[0]) - 1

    def _column_min(self) -> tuple[float, int]:
        """The cheapest partial explanation, and how many keystrokes it covers."""
        values = grid_values(self.column)
        best_value, best_index = values[0], 0
        for i in range(1, len(values)):
            if values[i] < best_value:
                best_value, best_index = values[i], i
        return best_value, best_index

    def cost_bound(self) -> float:
        """Lower bound on the channel cost of every string reachable below.

        Two ways a descendant can be cheap. It can beat the current column --
        but a column minimum never decreases as the candidate grows, so
        ``min(column)`` bounds that. Or it can inherit a match this node has
        *already* achieved at some prefix, since everything after that prefix
        is free tail rather than error -- so ``best_cost`` bounds that. A
        descendant cannot do better than both.
        """
        return min(self.best_cost, self._column_min()[0])

    def bound(self) -> float:
        """Upper bound on the score of every string reachable below here.

        Sound: used for pruning and for the stopping rule, both of which must
        never discard a candidate that could have won.
        """
        return self.logprob - self.cost_bound()

    def keystrokes_owed(self) -> int:
        """Keystrokes not explained by the cheapest partial alignment.

        Reported for diagnostics only. On its own it is a poor signal once
        gaps are cheap -- see :meth:`priority`.
        """
        column_min, covered = self._column_min()
        if self.best_cost <= column_min:
            return 0
        return self.query_len - covered

    def priority(self, progress_penalty: float) -> float:
        """Search order. Deliberately *not* the sound bound.

        Pure best-first on ``bound()`` collapses to breadth-first, because
        extending a path can only lower its score -- so a one-token path like
        "I" always outranks the ten-token path that actually answers the
        query. Against "I will gt bck" that is fatal: the channel makes a
        candidate explain *every* keystroke, so short candidates are never
        emitted at all and the search returns nothing while looking busy.

        So a node is ordered by cost already incurred *plus* an estimate of
        what it still owes: ``progress_penalty`` nats for each keystroke not
        yet explained, minimised over every alignment state.

        Minimising the sum is the whole trick, and taking the cheapest
        alignment first and penalising it afterwards is not the same thing.
        Once gaps became cheap, the cheapest partial alignment of nearly every
        node was "open one gap and explain nothing at all", so the count of
        unexplained keystrokes read as the full query everywhere, the estimate
        became a constant, and the search lost its sense of progress
        completely -- "th wthr hs bn" fell from 146 candidates to 4. Scored
        together, explaining nothing costs its 2 nats plus the whole query's
        worth of penalty, and a real seven-keystroke alignment wins.

        The estimate can be wrong, which is why it is confined to the heap
        order: pruning and stopping keep using the admissible ``bound()``, so
        a bad estimate can slow the search down but never make it drop a
        candidate it should have kept. A penalty of 0 restores plain A*.
        """
        values = grid_values(self.column)
        last = len(values) - 1
        estimate = self.best_cost  # already explains everything typed
        for covered, cost in enumerate(values):
            owed = cost + progress_penalty * (last - covered)
            if owed < estimate:
                estimate = owed
        return self.logprob - estimate


@dataclass
class _Merged:
    """Every token spelling that decodes to the same string."""

    total: float = NEG_INF
    best_logprob: float = NEG_INF
    raw: str = ""
    #: The winning path's text *including* its terminator. Rescoring prices
    #: this rather than ``raw``: the walk's token paths all end in a
    #: terminator, so pricing the bare word would price a strict prefix of
    #: them and add overlapping probability to the same total.
    terminated: str = ""
    tokens: tuple[int, ...] = ()
    cost: float = 0.0
    consumed: int = 0
    keystrokes: int = 0
    n_paths: int = 0
    #: Every token spelling already counted, so no probability is added twice.
    paths: set[tuple[int, ...]] = field(default_factory=set)

    def add(
        self,
        logprob: float,
        raw: str,
        tokens: tuple[int, ...],
        terminated: str | None = None,
    ) -> None:
        """Merge one token spelling of this string.

        Spellings are disjoint events -- distinct token sequences, none a
        prefix of another -- so their probabilities add. Keeping the set of
        spellings already counted is what enforces that.
        """
        if tokens in self.paths:
            return
        self.paths.add(tokens)
        self.total = logsumexp((self.total, logprob))
        self.n_paths += 1
        if logprob > self.best_logprob:
            self.best_logprob = logprob
            self.raw = raw
            if terminated is not None:
                self.terminated = terminated
            self.tokens = tokens


def _key_of(text: str) -> str:
    """The word or phrase a text has completed, terminator removed."""
    return text.rstrip(_TERMINATOR_STR).strip()


def _emission_key(parent_text: str, text: str) -> str | None:
    """The string this step just finished, or None if it finished nothing.

    The terminator is dropped, so " clothes ", " clothes." and " clothes!"
    report the same completed word and their probabilities *sum*: the wanted
    quantity is "P(the word is clothes)", marginalised over whatever
    punctuation follows, not one number per punctuation mark.

    Emitting requires that this step actually advanced the key, which stops a
    path reporting the same word twice while walking a run of terminators.
    """
    if not text or text[-1] not in TERMINATORS:
        return None
    key = _key_of(text)
    if not key:
        return None
    if parent_text and parent_text[-1] in TERMINATORS and _key_of(parent_text) == key:
        return None
    return key


def _grow(
    parent: _Node, text: str, query: str, costs: ChannelCosts
) -> tuple[Grid, float, int]:
    """Carry the channel grid forward from ``parent`` to the longer ``text``."""
    old = parent.text.lstrip()
    new = text.lstrip()
    if new.startswith(old):
        column, best, consumed = parent.column, parent.best_cost, parent.best_consumed
        added, base = new[len(old) :], len(old)
    else:
        # The parent's text was not a prefix of the child's, which byte-level
        # BPE can do when a token completes a character. Rebuild rather than
        # let the grid describe a different string than the text.
        column = initial_column(len(query), costs)
        best, consumed = grid_values(column)[-1], 0
        added, base = new, 0
    m = len(query)
    for offset, ch in enumerate(added, start=1):
        column = push_candidate_char(column, query, ch, costs)
        # The cost of explaining *every* keystroke with the text so far.
        full = min(column[0][m], column[1][m])
        if full < best:
            best, consumed = full, base + offset
    return column, best, consumed


def _child(
    parent: _Node, tokens: tuple[int, ...], data: bytes, logprob: float,
    query: str, costs: ChannelCosts,
) -> _Node:
    text = data.decode("utf-8", errors="replace")
    if text.endswith(_PARTIAL):
        # Half a character: not a string yet, so the channel does not advance.
        return _Node(
            tokens=tokens, logprob=logprob, data=data, text=parent.text,
            column=parent.column, best_cost=parent.best_cost,
            best_consumed=parent.best_consumed,
        )
    column, best, consumed = _grow(parent, text, query, costs)
    return _Node(
        tokens=tokens, logprob=logprob, data=data, text=text,
        column=column, best_cost=best, best_consumed=consumed,
    )


def _boundary_ids(
    lm: LanguageModel, node: _Node, query: str, limit: int
) -> tuple[int, ...]:
    """Words that could start where the keystrokes have not been explained yet.

    Anchoring only the first word is not enough. "helhay" means "hello how are
    you": seeding gets the search to "Hello", and then " how are you" has to
    beat a one-token " everyone" on prior three times over, which it never
    does -- so the abbreviation dies one word in.

    At each word boundary the keystrokes that remain unexplained are looked up
    in the vocabulary again, and the matches are kept in play regardless of
    their rank. An abbreviation is a sequence of word-initial fragments, so
    this simply re-anchors on each of them in turn. It costs a bisect per node
    and a gather per row, not a forward pass.
    """
    if not query:
        return ()
    if node.text and node.text[-1] not in " \t\n":
        return ()  # mid-word: the next token continues it, not a new word
    remaining = query[node._column_min()[1] :].lstrip()
    if not remaining:
        return ()
    # The fragment length is unknowable here: in "helhay" the "h" of "hay" is
    # a whole word ("how"), yet "hay" is also a word. Guessing the longest
    # match picks "hay" and loses "how" for good. So every plausible fragment
    # is proposed and the model ranks them -- scoring a proposal is a gather
    # from a row already computed, and only the best few become children.
    ids: list[int] = []
    for cut in range(1, min(len(remaining), 4) + 1):
        ids.extend(lm.tokens_with_prefix(remaining[:cut], limit))
    return tuple(dict.fromkeys(ids))


def _top_k_complete(
    merged: dict[str, _Merged], k: int, bound: float, margin: float
) -> bool:
    if len(merged) < k:
        return False
    kth = heapq.nlargest(k, (m.total - m.cost for m in merged.values()))[-1]
    return bound <= kth + math.log(margin)


def _seed_token_paths(
    lm: LanguageModel, seeds: Sequence[str]
) -> list[tuple[int, ...]]:
    """Token paths to start from, given the literal texts typed.

    Seeding the raw keystrokes alone does not work, and the reason is
    tokenization rather than probability. Measured on this model, " apricot"
    is spelled [" apr", "icot"] while the partial " aprico" is [" apr",
    "ico"] -- the natural route to the finished word does not pass through
    the partial word's token path at all, and from "ico" the model's likeliest
    continuations are "les" and "es"; "t" is not in its top eight. A path that
    can only be completed into something nobody meant is worse than useless.

    So each seed also contributes its longest *token-aligned* prefix, dropping
    the trailing partial token. " apr" is a token the model itself would
    write, its natural continuation is "icot", and its channel cost bound is
    zero because it prefix-matches the keystrokes -- so it competes on an
    honest prior instead of a spelling artefact. The full text is kept too,
    which is what guarantees that literally typing something always leaves it
    available.
    """
    paths: list[tuple[int, ...]] = []
    for text in seeds:
        tokens = tuple(lm.encode(text))
        if not tokens:
            continue
        for candidate in (tokens, tokens[:-1]):
            if candidate and candidate not in paths:
                paths.append(candidate)
    return paths


def _vocabulary_nodes(
    lm: LanguageModel, prefix: tuple[int, ...], seeds: Sequence[str],
    root: _Node, query: str, costs: ChannelCosts, limit: int, top: int,
) -> list[_Node]:
    """Whole words from the vocabulary that begin with what was typed.

    Walking the tree cannot find a word the model never proposes, and a
    single-token word can be a good guess while sitting thousands of places
    down the distribution: after the default preamble "Hello" scores -14.2
    against "Here" at -11.6, under three nats apart, yet far outside the
    top-64 the search expands. Typing "hel" would then offer every "Here ..."
    and never "Hello" -- and adding the "l" would make it *worse*, since the
    letter can only be charged as a slip.

    The vocabulary knows the word. Candidates are found by prefix, priced by
    a single forward pass whose cost does not depend on how many were found,
    and the best few are handed to the search as starting paths, where they
    compete on the same posterior as everything else.
    """
    if not prefix or top < 1:
        return []
    wanted, ids = [s.strip().lower() for s in seeds if s.strip()], []
    for text in dict.fromkeys(wanted):
        # Back off to the longest prefix that is actually the start of a word.
        # Whole keystrokes rarely are once abbreviation gets going: "helhay"
        # means "hello how are you" and begins no token at all, while its
        # first three characters begin "hello", "help" and "held". Anchoring
        # on the first word is what gives the search somewhere real to start;
        # the channel then explains the rest of the keystrokes as the cheap
        # gaps they are.
        for cut in range(len(text), 1, -1):
            found = lm.tokens_with_prefix(text[:cut], limit)
            if found:
                ids.extend(found)
                break
    ids = list(dict.fromkeys(ids))
    if not ids:
        return []
    ranked = sorted(zip(ids, lm.token_logprobs(prefix, ids)), key=lambda kv: -kv[1])
    return [
        _child(root, (tid,), lm.token_bytes(tid), logprob, query, costs)
        for tid, logprob in ranked[:top]
    ]


def _seed_nodes(
    lm: LanguageModel, prefix: tuple[int, ...], seeds: Sequence[str],
    root: _Node, query: str, costs: ChannelCosts,
) -> list[_Node]:
    """Turn literal texts into starting paths, priced by a real forward pass."""
    paths = _seed_token_paths(lm, seeds)
    if not paths or not prefix:
        return []
    logprobs = lm.sequence_logprobs([(prefix, toks) for toks in paths])
    nodes = []
    for toks, lp in zip(paths, logprobs):
        data = b"".join(lm.token_bytes(t) for t in toks)
        nodes.append(_child(root, toks, data, lp, query, costs))
    return nodes


def _rescore_canonical(
    lm: LanguageModel,
    prefix: tuple[int, ...],
    merged: dict[str, _Merged],
    limit: int,
) -> int:
    """Add each top candidate's canonical spelling to its probability.

    The walk finds a string by whichever token path it happened to take, and
    a seeded branch in particular spells a *partial* word. Measured on this
    model, " aprico" scores -16.49 while " apricot" scores -8.30 -- an 8-nat
    penalty that is an artefact of an unnatural token split, not a statement
    about apricots. Pricing the natural spelling and merging it in removes
    that artefact, and helps every other candidate too, since no walk
    enumerates all of a string's spellings.

    Returns how many were re-priced.
    """
    if not prefix or limit < 1:
        return 0
    ranked = sorted(merged.values(), key=lambda m: -(m.total - m.cost))[:limit]
    items: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    targets: list[tuple[_Merged, tuple[int, ...]]] = []
    for m in ranked:
        canonical = tuple(lm.encode(m.terminated or m.raw))
        if not canonical or canonical in m.paths:
            continue
        items.append((prefix, canonical))
        targets.append((m, canonical))
    if not items:
        return 0
    for (m, canonical), logprob in zip(targets, lm.sequence_logprobs(items)):
        m.add(logprob, m.raw, canonical, terminated=m.terminated)
    return len(items)


def _collect(merged: dict[str, _Merged], k: int) -> list[Candidate]:
    """Rank what has been found so far."""
    candidates = [
        Candidate(
            text=key, raw=m.raw, logprob=m.total, cost=m.cost,
            consumed=m.consumed, keystrokes=m.keystrokes,
            n_paths=m.n_paths, tokens=m.tokens,
        )
        for key, m in merged.items()
    ]
    candidates.sort(key=lambda c: (-c.score, c.text))
    return candidates[:k]


def predict(
    lm: LanguageModel,
    prefix_ids: Sequence[int],
    query: str = "",
    config: PredictConfig | None = None,
    costs: ChannelCosts | None = None,
    on_progress: Callable[[PredictStats], None] | None = None,
    seeds: Sequence[str] = (),
    on_candidates: Callable[[list[Candidate], PredictStats], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> tuple[list[Candidate], PredictStats]:
    """Rank the strings the typist most likely meant.

    ``prefix_ids`` is the tokenized context to continue. ``query`` is the raw
    keystrokes typed since the last commit; empty means "no evidence yet",
    which reduces the search to a plain most-probable-continuation walk.
    ``seeds`` are literal texts to insert as starting paths -- normally the
    keystrokes themselves, supplied by the caller because only it knows
    whether a leading space belongs there.

    ``on_candidates`` receives the ranking so far every
    ``PredictConfig.publish_every`` rounds. Searching longer finds more and
    better phrases, but a typist should not have to wait for the end of it to
    see anything, so results are published as they are found. ``should_stop``
    is polled once per round and abandons the search: a keystroke that the
    current pool cannot explain matters more than finishing a decode that is
    already out of date.
    """
    started = time.monotonic()
    cfg = config or PredictConfig()
    ch_costs = costs or ChannelCosts()
    prefix = tuple(prefix_ids)
    budget = ch_costs.budget(len(query))
    stats = PredictStats()
    merged: dict[str, _Merged] = {}
    pushed: set[tuple[int, ...]] = set()

    root_column = initial_column(len(query), ch_costs)
    root = _Node(
        tokens=(), logprob=0.0, data=b"", text="",
        column=root_column, best_cost=grid_values(root_column)[-1],
        best_consumed=0,
    )
    tiebreak = 0
    penalty = cfg.progress_penalty
    frontier: list[tuple[float, int, _Node]] = [
        (-root.priority(penalty), tiebreak, root)
    ]
    pushed.add(())
    stats.nodes_pushed = 1

    if cfg.seed_query and seeds:
        model_started = time.monotonic()
        seed_nodes = _seed_nodes(lm, prefix, seeds, root, query, ch_costs)
        seed_nodes += _vocabulary_nodes(
            lm, prefix, seeds, root, query, ch_costs,
            cfg.seed_vocab_limit, cfg.seed_vocab_top,
        )
        stats.seconds_model += time.monotonic() - model_started
        for node in seed_nodes:
            if node.tokens in pushed:
                continue
            tiebreak += 1
            heapq.heappush(frontier, (-node.priority(penalty), tiebreak, node))
            pushed.add(node.tokens)
            stats.nodes_pushed += 1
            stats.seeded += 1

    while frontier and stats.rounds < cfg.max_rounds:
        if stats.expansions >= cfg.max_expansions:
            break
        if should_stop is not None and should_stop():
            stats.interrupted = True
            break
        # The heap is ordered by priority, which is not a bound; the stopping
        # proof needs the best true bound still in the frontier.
        best_bound = max(node.bound() for _, _, node in frontier)
        if _top_k_complete(merged, cfg.k, best_bound, cfg.stop_margin):
            stats.complete_top_k = True
            break

        batch: list[_Node] = []
        while frontier and len(batch) < cfg.batch_size:
            batch.append(heapq.heappop(frontier)[2])

        extras = [
            _boundary_ids(lm, node, query, cfg.boundary_vocab_limit)
            for node in batch
        ]
        model_started = time.monotonic()
        tops = lm.top_next(
            [prefix + n.tokens for n in batch],
            top_k=cfg.child_top_k,
            top_p=cfg.child_top_p,
            extra_ids=extras,
            extra_keep=cfg.boundary_vocab_top,
        )
        stats.seconds_model += time.monotonic() - model_started
        stats.rounds += 1
        stats.expansions += len(batch)

        # Anything that cannot reach the current k-th best score is not worth
        # pushing: its bound already says no descendant of it can either. This
        # is the same test that proves the search may stop, applied per child,
        # and it adapts as better candidates are found instead of guessing an
        # absolute floor.
        cutoff = NEG_INF
        if len(merged) >= cfg.k:
            kth = heapq.nlargest(
                cfg.k, (m.total - m.cost for m in merged.values())
            )[-1]
            cutoff = kth + math.log(cfg.stop_margin)

        for node, top in zip(batch, tops):
            for tid, tlp in zip(top.token_ids, top.logprobs):
                child_lp = node.logprob + tlp
                if child_lp < cfg.min_logprob:
                    stats.pruned_by_prior += 1
                    continue
                if tid in lm.special_token_ids or tid == lm.eos_token_id:
                    continue

                tokens = node.tokens + (tid,)
                if tokens in pushed:
                    stats.duplicates += 1
                    continue
                child = _child(
                    node, tokens, node.data + lm.token_bytes(tid), child_lp,
                    query, ch_costs,
                )

                key = _emission_key(node.text, child.text)
                if key is not None:
                    # A candidate need only account for *part* of what has
                    # been typed: the rest is the next suggestion's job, not a
                    # mistake. Charging it as error made every correct prefix
                    # of a long shorthand unofferable.
                    offered, covered = partial_cost(
                        child.column, len(query), ch_costs
                    )
                else:
                    offered, covered = 0.0, 0
                # The budget bounds *errors*, and keystrokes left for later
                # are not errors, so they are taken back out before the test.
                errors = offered - ch_costs.tail_charge(len(query) - covered)
                if (
                    key is not None
                    and errors <= ch_costs.budget(covered)
                    and (covered > 0 or not query)
                ):
                    entry = merged.setdefault(key, _Merged())
                    entry.cost = offered
                    entry.keystrokes = covered
                    entry.consumed = min(child.best_consumed, len(key))
                    entry.add(
                        child_lp,
                        child.text.rstrip(_TERMINATOR_STR),
                        tokens,
                        terminated=child.text,
                    )

                if any(stop in child.text for stop in _HARD_STOP):
                    continue
                # A branch that can no longer explain the keystrokes within
                # the error budget is not decoded further. This asks the same
                # bound as the search priority: a node that already matched at
                # some prefix stays alive, because its extra characters are
                # prediction, not error.
                if child.cost_bound() > budget:
                    stats.pruned_by_channel += 1
                    continue
                if child.bound() < cutoff:
                    stats.pruned_by_cutoff += 1
                    continue
                if len(tokens) >= cfg.max_tokens or len(child.text) >= cfg.max_chars:
                    stats.truncated += 1
                    continue

                tiebreak += 1
                heapq.heappush(frontier, (-child.priority(penalty), tiebreak, child))
                pushed.add(tokens)
                stats.nodes_pushed += 1

        stats.distinct = len(merged)
        if on_progress is not None:
            on_progress(stats)
        if on_candidates is not None and stats.rounds % cfg.publish_every == 0:
            stats.seconds_total = time.monotonic() - started
            on_candidates(_collect(merged, cfg.k), stats)

    stats.exhausted = not frontier
    stats.frontier_bound = (
        max(node.bound() for _, _, node in frontier) if frontier else NEG_INF
    )

    model_started = time.monotonic()
    stats.rescored = _rescore_canonical(lm, prefix, merged, cfg.rescore_top)
    stats.seconds_model += time.monotonic() - model_started

    stats.distinct = len(merged)
    stats.seconds_total = time.monotonic() - started
    return _collect(merged, cfg.k), stats
