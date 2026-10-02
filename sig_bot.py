"""
SIG Predictions Cup trading bot: single-file version.

HOW TO RUN
  1. Fill in the three settings in the "YOUR SETTINGS" block below.
  2. In a VS Code terminal:   python sig_bot.py
     (Mac/Linux may need:     python3 sig_bot.py)
  3. Leave LIVE_TRADING = False first. It then only PRINTS the orders it
     would place. Once those look sensible, set it to True to trade.
  4. Ctrl+C stops it. Resting quotes stay on the book until you cancel
     them on the site, or until the next run's first cycle cancels them.

No pip installs needed: standard library only (Python 3.9+).

WHAT IT DOES (every cycle, default 60 s)
  1. Arbitrage: in mutually exclusive markets, buy every YES if the asks sum
     below 1, or every NO if the bids sum above 1. Locked-in profit.
  2. Taking: estimate a fair probability from the order book, correcting for
     favourite-longshot bias, and buy when the book is off by more than 6c.
     Sized with quarter-Kelly.
  3. Market making: rest a bid below and an ask above fair value on wide
     books, skewed against inventory and widened when prices move fast.
  Risk limits: max 8% of equity lost on one contract, 15% on one market,
  10% always kept in cash, stop if equity falls 40% below its peak.
"""

# =============================================================================
# YOUR SETTINGS: change these
# =============================================================================

# Your API key. Super Market site -> My Profile -> API Keys -> Create New API Key.
# Tick the "read" AND "trade" scopes. Copy the key straight away (shown only once).
API_KEY = "PASTE_YOUR_API_KEY_HERE"            # <-- REPLACE

# The tournament's slug. Leave it as-is and run the script once: it prints
# every tournament you can access with its slug. Copy the right one here.
TOURNAMENT_SLUG = "PASTE_TOURNAMENT_SLUG_HERE"  # <-- REPLACE

# False = dry run (prints orders, places nothing). True = real trading.
LIVE_TRADING = False                           # <-- set to True when ready

# Optional tuning
LONGSHOT_GAMMA = 1.15      # favourite-longshot correction (1.0 = off)
CYCLE_SECONDS = 60         # how often to re-quote (lower = fresher quotes, more API calls)
MAX_DRAWDOWN = 0.40        # stop if equity falls this far below its peak
MAX_BOOKS_PER_CYCLE = 40   # order books read per cycle (rate limit: 100 reads/min)
BASE_URL = "https://www.thesuper.market/api/v1"

# =============================================================================
# Everything below is the bot. You don't need to edit it.
# =============================================================================

import json
import math
import os
import random
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Tuple

DEFAULT_BASE = BASE_URL


# =============================================================================
# Strategy parameters
# =============================================================================

# Exchange rules from the Super Market API spec.
TICK = 0.005          # limit prices must sit on a 0.005 grid
MIN_PRICE = 0.005     # lowest legal limit price
MAX_PRICE = 0.995     # highest legal limit price


@dataclass
class StrategyParams:
    # ---------------- fair value model ----------------
    # Fair value starts as a blend of the order-book price and an EWMA of
    # recent trade prices. The book reacts quickly; the EWMA filters out
    # one-off prints but lags real moves, so it gets a small weight.
    # The backtest found that any real weight on the lagging EWMA loses money
    # (it fades moves that carry information), so it is only a fallback for
    # books with no two-sided quote.
    micro_weight: float = 1.0        # weight of the book price vs the trade EWMA
    imbalance_weight: float = 0.0    # 0 = plain mid, 1 = full size-weighted microprice
    ewma_alpha: float = 0.3          # weight of the newest trade price in the EWMA

    # Favourite-longshot correction. Prediction-market crowds usually overpay
    # for longshots and underpay for favourites. We undo that with a power
    # transform: q_k ∝ p_k ** gamma (for a binary this equals
    # logit(q) = gamma * logit(p)). gamma = 1 switches the correction off.
    # Fit gamma on real settled markets with ``history.py calibrate``.
    longshot_gamma: float = 1.15

    # ---------------- 1. arbitrage ----------------
    # For a mutually exclusive basket, buy every YES when the asks sum below 1,
    # or every NO when the bids sum above 1. The payout is locked in whatever
    # happens. arb_min_profit is the minimum locked-in profit per basket.
    arb_min_profit: float = 0.005
    arb_max_sets: int = 2000         # cap on baskets bought per decision

    # ---------------- 2. taking (directional value bets) ----------------
    # Cross the spread only when the model's fair value beats the price by at
    # least min_take_edge. The buffer covers error in the fair value model.
    # The synthetic backtest found pure taking roughly breakeven at 0.03-0.04,
    # so it is kept for large dislocations only.
    min_take_edge: float = 0.06
    kelly_fraction: float = 0.25     # quarter-Kelly: most of the growth, far less variance

    # ---------------- 3. market making ----------------
    mm_enabled: bool = True
    mm_half_spread: float = 0.01     # quote at least this far from fair value
    # Volatility-scaled spread: half-spread = max(mm_half_spread, mm_vol_mult * vol),
    # where vol is an EWMA of how far fair value moves between decisions. When
    # prices move a lot between our refreshes, a resting quote is a free option
    # for whoever trades next, so we stand further back.
    mm_vol_mult: float = 1.5
    vol_alpha: float = 0.2
    mm_min_book_spread: float = 0.015  # skip books already tighter than this
    mm_size_frac: float = 0.01       # each quote risks about 1% of equity
    mm_inventory_skew: float = 0.06  # max fair-value shift when inventory is at its limit
    mm_stop_hours: float = 6.0       # stop quoting this close to settlement (news risk)

    # ---------------- risk limits ----------------
    max_exchange_risk_frac: float = 0.08  # max loss on any single contract, as a share of equity
    max_market_risk_frac: float = 0.15    # max loss on any single market (all its outcomes)
    cash_reserve_frac: float = 0.10       # always keep this share of equity in cash
    min_order_qty: int = 1

    # ---------------- component switches (used for ablation tests) ----------------
    arb_enabled: bool = True
    take_enabled: bool = True

# =============================================================================
# Strategy: the decision logic
# =============================================================================

Book = List[Tuple[float, float]]  # [(price, quantity), ...] YES-normalised


# ----------------------------------------------------------------------------
# Data passed into the strategy
# ----------------------------------------------------------------------------

@dataclass
class ExchangeView:
    """One tradable contract and its current order book."""
    exchange_id: str
    bids: Book            # YES bids, best (highest) first
    asks: Book            # YES asks, best (lowest) first
    last_price: Optional[float] = None


@dataclass
class GroupView:
    """A set of contracts priced together.

    A standalone binary market is a group of one. A multi-outcome market whose
    outcomes are mutually exclusive (``exclusive``) is a basket, and if exactly
    one outcome must win it is also ``exhaustive``. Independent multi-outcome
    markets (for example imported range outcomes) must be passed as one group
    per exchange, because basket arbitrage does not apply to them.
    """
    group_id: str
    market_id: str
    exchanges: List[ExchangeView]
    exclusive: bool = False
    exhaustive: bool = False
    hours_to_settle: float = float("inf")


@dataclass
class Position:
    """Holdings in one exchange. ``*_cost`` is cash paid, which is also the
    maximum we can lose on that side, because a contract never pays below 0."""
    yes_qty: float = 0.0
    yes_cost: float = 0.0
    no_qty: float = 0.0
    no_cost: float = 0.0

    @property
    def risk(self) -> float:
        return self.yes_cost + self.no_cost


@dataclass
class Portfolio:
    cash: float
    equity: float                     # cash + mark-to-market value of positions
    positions: Dict[str, Position] = field(default_factory=dict)


@dataclass
class OrderIntent:
    """An order the strategy wants placed. ``price`` is side-relative, matching
    the API: a NO buy at 0.30 is ``side='no', price=0.30`` (YES-equivalent 0.70)."""
    exchange_id: str
    side: str                 # 'yes' | 'no'
    price: float
    qty: int
    kind: str                 # 'arb' | 'take' | 'quote'
    basket_id: Optional[str] = None   # arb legs sharing an id go into one atomic multi-leg order

    @property
    def cash_needed(self) -> float:
        return self.qty * self.price


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------

def floor_tick(p: float) -> float:
    return round(math.floor(p / TICK + 1e-9) * TICK, 3)


def ceil_tick(p: float) -> float:
    return round(math.ceil(p / TICK - 1e-9) * TICK, 3)


def clamp_price(p: float) -> float:
    return min(MAX_PRICE, max(MIN_PRICE, p))


def book_price(bids: Book, asks: Book, imbalance_weight: float) -> Optional[float]:
    """Mid price, optionally leaned toward the microprice.

    The microprice is the size-weighted mid. If the bid is much bigger than the
    ask, the next trade is more likely to lift the ask, so the price leans
    toward it. That signal is real on busy books but pure noise on thin ones,
    and a noisy fair value makes the taker chase phantom edges. So the lean is
    weighted, and the backtest decides how much to trust it.
    """
    if not bids or not asks:
        return None
    (b, bq), (a, aq) = bids[0], asks[0]
    mid = (a + b) / 2
    if bq + aq <= 0:
        return mid
    micro = (b * aq + a * bq) / (bq + aq)
    return mid + imbalance_weight * (micro - mid)


def kelly_fraction(win_prob: float, price: float) -> float:
    """Kelly-optimal share of bankroll to stake on a contract that costs
    ``price`` and pays 1 with probability ``win_prob``."""
    if price <= 0 or price >= 1:
        return 0.0
    return max(0.0, (win_prob - price) / (1.0 - price))


# ----------------------------------------------------------------------------
# Strategy
# ----------------------------------------------------------------------------

class Strategy:
    def __init__(self, params: Optional[StrategyParams] = None):
        self.p = params or StrategyParams()
        self._ewma: Dict[str, float] = {}   # per-exchange EWMA of traded prices
        self._last_fair: Dict[str, float] = {}
        self._vol: Dict[str, float] = {}    # per-exchange EWMA of |change in fair value| per decision

    # ---- state updates --------------------------------------------------

    def observe_price(self, exchange_id: str, price: Optional[float]) -> None:
        """Feed a trade (or last-traded) price into the exchange's EWMA."""
        if price is None:
            return
        prev = self._ewma.get(exchange_id)
        a = self.p.ewma_alpha
        self._ewma[exchange_id] = price if prev is None else a * price + (1 - a) * prev

    # ---- fair value -----------------------------------------------------

    def raw_probability(self, ex: ExchangeView) -> Optional[float]:
        """The crowd's implied probability: book price blended with the trade EWMA."""
        micro = book_price(ex.bids, ex.asks, self.p.imbalance_weight)
        ewma = self._ewma.get(ex.exchange_id, ex.last_price)
        if micro is None:
            # One-sided book. The touch only bounds the price: with asks alone
            # the probability is somewhere in [0, ask]. Using the ask itself
            # badly overstates longshots, which in a basket also drags every
            # other outcome's fair value down. Use the EWMA if it falls inside
            # the bound, otherwise the middle of the bound.
            if not ex.bids and not ex.asks:
                return ewma
            lo = ex.bids[0][0] if ex.bids else 0.0
            hi = ex.asks[0][0] if ex.asks else 1.0
            if ewma is not None and lo <= ewma <= hi:
                return ewma
            return (lo + hi) / 2
        if ewma is None:
            return micro
        w = self.p.micro_weight
        return w * micro + (1 - w) * ewma

    def fair_values(self, group: GroupView) -> Dict[str, float]:
        """Fair probability for every exchange in a group.

        1. Take each contract's raw implied probability.
        2. In an exhaustive basket the true probabilities sum to 1, so divide
           out the overround (the bookmaker margin baked into the prices).
           This needs a two-sided book on every outcome; one bad estimate
           would otherwise shift every fair value in the basket.
        3. Apply the longshot correction q_k ∝ p_k ** gamma. A binary is
           treated as the two-outcome basket {p, 1 - p}.
        """
        g = self.p.longshot_gamma
        raws = {ex.exchange_id: self.raw_probability(ex) for ex in group.exchanges}

        two_sided = all(ex.bids and ex.asks for ex in group.exchanges)
        if group.exhaustive and len(group.exchanges) >= 2 and two_sided:
            ps = {k: min(max(v, 1e-4), 1 - 1e-4) for k, v in raws.items()}
            weights = {k: v ** g for k, v in ps.items()}
            z = sum(weights.values())
            return {k: w / z for k, w in weights.items()}

        out: Dict[str, float] = {}
        for k, v in raws.items():
            if v is None:
                continue
            v = min(max(v, 1e-4), 1 - 1e-4)
            yes, no = v ** g, (1 - v) ** g
            out[k] = yes / (yes + no)
        return out

    # ---- main entry point -----------------------------------------------

    def decide(self, groups: List[GroupView], pf: Portfolio) -> List[OrderIntent]:
        """Return every order to place this cycle. The caller cancels resting
        quotes first, so the quotes here are a complete replacement set."""
        p = self.p
        intents: List[OrderIntent] = []
        equity = max(pf.equity, 1e-9)

        # Spendable cash. Every resting buy locks collateral, so quotes count too.
        budget = [pf.cash - p.cash_reserve_frac * equity]

        # Work on copies of the books so that liquidity we take in one stage
        # cannot be counted again in a later stage.
        books = {ex.exchange_id: ([list(l) for l in ex.bids], [list(l) for l in ex.asks])
                 for grp in groups for ex in grp.exchanges}

        # Exposure: what we hold plus what we have planned this cycle.
        exch_market = {ex.exchange_id: grp.market_id for grp in groups for ex in grp.exchanges}
        pos = {k: Position(v.yes_qty, v.yes_cost, v.no_qty, v.no_cost) for k, v in pf.positions.items()}
        market_risk: Dict[str, float] = {}
        for ex_id, ps in pos.items():
            m = exch_market.get(ex_id)
            if m is not None:
                market_risk[m] = market_risk.get(m, 0.0) + ps.risk

        def position(ex_id: str) -> Position:
            return pos.setdefault(ex_id, Position())

        def risk_room(ex_id: str) -> float:
            """How much more we may lose on this exchange and its market."""
            ps = position(ex_id)
            m = exch_market[ex_id]
            return max(0.0, min(p.max_exchange_risk_frac * equity - ps.risk,
                                p.max_market_risk_frac * equity - market_risk.get(m, 0.0)))

        def book_fill(ex_id: str, side: str, qty: int, price: float, kind: str) -> None:
            """Record a planned buy in exposure, budget and the book copy.
            Buying the opposite side of what we hold nets off: a YES/NO pair
            is worth exactly 1, so risk goes down, not up."""
            ps = position(ex_id)
            m = exch_market[ex_id]
            held_qty = ps.no_qty if side == "yes" else ps.yes_qty
            netted = min(qty, held_qty)
            new = qty - netted
            if side == "yes":
                if netted:
                    ps.no_cost -= ps.no_cost * netted / ps.no_qty
                    ps.no_qty -= netted
                ps.yes_qty += new
                ps.yes_cost += new * price
            else:
                if netted:
                    ps.yes_cost -= ps.yes_cost * netted / ps.yes_qty
                    ps.yes_qty -= netted
                ps.no_qty += new
                ps.no_cost += new * price
            market_risk[m] = sum(position(e).risk for e, mm in exch_market.items() if mm == m)
            budget[0] -= qty * price
            bids, asks = books[ex_id]
            # A YES buy lifts asks; a NO buy at price n hits YES bids at 1 - n.
            levels, yes_px = (asks, price) if side == "yes" else (bids, round(1 - price, 3))
            remaining = qty
            for lvl in levels:
                if remaining <= 0:
                    break
                if (side == "yes" and lvl[0] > yes_px + 1e-9) or (side == "no" and lvl[0] < yes_px - 1e-9):
                    break
                take = min(lvl[1], remaining)
                lvl[1] -= take
                remaining -= take
            levels[:] = [l for l in levels if l[1] > 0]

        # ================= 1. ARBITRAGE =================
        if p.arb_enabled:
            for grp in groups:
                if not grp.exclusive or len(grp.exchanges) < 2:
                    continue
                n = len(grp.exchanges)
                ids = [ex.exchange_id for ex in grp.exchanges]

                # (a) YES basket. Only valid when exactly one outcome must win.
                if grp.exhaustive:
                    self._arb_walk(grp, ids, books, budget, intents, book_fill,
                                   side="yes", payoff=1.0)
                # (b) NO basket. Valid for any exclusive set: at most one YES
                # wins, so at least n - 1 of our NO contracts pay out.
                self._arb_walk(grp, ids, books, budget, intents, book_fill,
                               side="no", payoff=float(n - 1))

        # Fair values are computed once and reused by taking and quoting.
        fair: Dict[str, float] = {}
        hours: Dict[str, float] = {}
        for grp in groups:
            fair.update(self.fair_values(grp))
            for ex in grp.exchanges:
                hours[ex.exchange_id] = grp.hours_to_settle
        # Track how much each fair value moves between decisions (feeds the quote width).
        for ex_id, q in fair.items():
            prev = self._last_fair.get(ex_id)
            if prev is not None:
                a = p.vol_alpha
                self._vol[ex_id] = a * abs(q - prev) + (1 - a) * self._vol.get(ex_id, abs(q - prev))
            self._last_fair[ex_id] = q

        # ================= 2. TAKING =================
        if p.take_enabled:
            # Visit the biggest mispricings first so they get the cash first.
            def best_edge(ex_id: str) -> float:
                bids, asks = books[ex_id]
                q = fair[ex_id]
                e1 = q - asks[0][0] if asks else -1
                e2 = bids[0][0] - q if bids else -1
                return max(e1, e2)

            for ex_id in sorted(fair, key=best_edge, reverse=True):
                if best_edge(ex_id) < p.min_take_edge:
                    break
                if not (books[ex_id][0] and books[ex_id][1]):
                    continue  # one-sided book: fair value too uncertain to bet on
                self._take(ex_id, fair[ex_id], books, budget, equity, intents,
                           book_fill, risk_room, position)

        # ================= 3. MARKET MAKING =================
        if p.mm_enabled:
            for ex_id, q in fair.items():
                if hours[ex_id] <= p.mm_stop_hours or budget[0] <= 0:
                    continue
                self._quote(ex_id, q, books, budget, equity, intents,
                            book_fill, risk_room, position)

        return [i for i in intents if i.qty >= p.min_order_qty]

    # ---- stage helpers --------------------------------------------------

    def _arb_walk(self, grp, ids, books, budget, intents, book_fill, side, payoff):
        """Buy complete baskets level by level while each extra basket still
        locks in at least ``arb_min_profit``.

        For each exchange we track the worst price reached. Each leg is then
        sent as one limit order at that price, so it sweeps the better levels
        too. Legs go into one atomic multi-leg request (shared basket_id).
        """
        p = self.p
        # Local copies, so that walking the book here does not double-count.
        local = {}
        for e in ids:
            bids, asks = books[e]
            if side == "yes":
                local[e] = [[a, q] for a, q in asks]               # buy YES at the ask
            else:
                local[e] = [[round(1 - b, 3), q] for b, q in bids]  # buy NO at 1 - bid
        filled = {e: 0 for e in ids}
        worst = {e: 0.0 for e in ids}
        sets, spent = 0, 0.0
        while sets < p.arb_max_sets:
            if any(not local[e] for e in ids):
                break
            cost = sum(local[e][0][0] for e in ids)
            if payoff - cost < p.arb_min_profit:
                break
            qty = int(min(min(local[e][0][1] for e in ids),
                          p.arb_max_sets - sets,
                          max(0.0, budget[0] - spent) / cost))
            if qty <= 0:
                break
            for e in ids:
                worst[e] = max(worst[e], local[e][0][0])
                filled[e] += qty
                local[e][0][1] -= qty
                if local[e][0][1] <= 0:
                    local[e].pop(0)
            sets += qty
            spent += qty * cost
        if sets == 0:
            return
        # book_fill charges each leg at its worst price, slightly more than the
        # true cost, which errs on the safe side.
        basket_id = f"{grp.group_id}:{side}"
        for e in ids:
            intents.append(OrderIntent(e, side, worst[e], filled[e], "arb", basket_id))
            book_fill(e, side, filled[e], worst[e], "arb")

    def _take(self, ex_id, q, books, budget, equity, intents, book_fill, risk_room, position):
        """Cross the spread on whichever side the fair value says is cheap.

        Walk the book one level at a time. Each level gets a fresh Kelly
        target, so as prices worsen the target shrinks and we stop on our own.
        """
        p = self.p
        bids, asks = books[ex_id]
        for side in ("yes", "no"):
            if side == "yes":
                levels = [(a, sz) for a, sz in asks]          # pay the ask
                win = q
            else:
                levels = [(round(1 - b, 3), sz) for b, sz in bids]  # pay 1 - bid for NO
                win = 1 - q
            total, worst = 0, 0.0
            for px, sz in levels:
                if win - px < p.min_take_edge:
                    break
                ps = position(ex_id)
                held_same = ps.yes_cost if side == "yes" else ps.no_cost
                target = p.kelly_fraction * kelly_fraction(win, px) * equity
                want_risk = min(target - held_same - total * px,
                                risk_room(ex_id) - total * px,
                                budget[0] - total * px)
                qty = int(min(sz, want_risk / px)) if want_risk > 0 else 0
                if qty <= 0:
                    break
                total += qty
                worst = px
            if total > 0:
                intents.append(OrderIntent(ex_id, side, worst, total, "take"))
                book_fill(ex_id, side, total, worst, "take")

    def _quote(self, ex_id, q, books, budget, equity, intents, book_fill, risk_room, position):
        """Rest a bid and an ask around fair value.

        * Never quote inside the half-spread of fair value, so every fill has
          positive expected value against our model. The half-spread widens
          with recent volatility, because a quote that is stale by the time it
          fills is mostly filled by people who know it is stale.
        * Improve the best bid or offer by at most one tick. Paying up beyond
          that just gives away edge. The exception is the side that reduces
          inventory, which may undercut by the skew amount.
        * Never cross the book: our orders must rest, not take.
        * Shift both quotes against inventory. When long, quote lower so that
          sells fill more often than buys.
        """
        p = self.p
        bids, asks = books[ex_id]
        best_bid = bids[0][0] if bids else None
        best_ask = asks[0][0] if asks else None
        if best_bid is not None and best_ask is not None and best_ask - best_bid < p.mm_min_book_spread:
            return

        ps = position(ex_id)
        cap = p.max_exchange_risk_frac * equity
        inventory = max(-1.0, min(1.0, (ps.yes_cost - ps.no_cost) / cap)) if cap > 0 else 0.0
        shift = p.mm_inventory_skew * inventory
        qs = q - shift
        half = max(p.mm_half_spread, p.mm_vol_mult * self._vol.get(ex_id, 0.0))
        # Normally we improve the touch by one tick at most. The side that
        # reduces inventory may undercut further, by the skew, to get out.
        extra_bid = max(0.0, -shift)   # short: bid more aggressively
        extra_ask = max(0.0, shift)    # long: offer more aggressively

        # ---- bid (buy YES) ----
        bid = floor_tick(qs - half)
        if best_bid is not None:
            bid = min(bid, floor_tick(best_bid + TICK + extra_bid))
        if best_ask is not None:
            bid = min(bid, round(best_ask - TICK, 3))
        bid = clamp_price(bid)

        # ---- ask (sell YES == buy NO at 1 - ask) ----
        ask = ceil_tick(qs + half)
        if best_ask is not None:
            ask = max(ask, ceil_tick(best_ask - TICK - extra_ask))
        if best_bid is not None:
            ask = max(ask, round(best_bid + TICK, 3))
        ask = clamp_price(ask)

        if bid >= ask:
            return

        # Size both quotes from the same pre-quote state. Each side is sized as
        # if it alone fills; the cash for both is still reserved.
        size_cash = p.mm_size_frac * equity
        room = risk_room(ex_id)
        no_px = round(1 - ask, 3)
        # Buying the side opposite our holding nets off, so it adds no risk.
        room_yes = room + ps.no_qty * bid
        room_no = room + ps.yes_qty * no_px
        qty_bid = int(min(size_cash, room_yes, max(0.0, budget[0])) / bid)
        qty_ask = int(min(size_cash, room_no, max(0.0, budget[0] - qty_bid * bid)) / no_px)

        if qty_bid > 0:
            intents.append(OrderIntent(ex_id, "yes", bid, qty_bid, "quote"))
            budget[0] -= qty_bid * bid
        if qty_ask > 0:
            intents.append(OrderIntent(ex_id, "no", no_px, qty_ask, "quote"))
            budget[0] -= qty_ask * no_px

# =============================================================================
# API client
# =============================================================================



class ApiError(Exception):
    def __init__(self, status: int, code: str, message: str, details: Any = None):
        super().__init__(f"{status} {code}: {message}")
        self.status, self.code, self.details = status, code, details


class TokenBucket:
    """Allows ``per_minute`` calls per rolling minute, refilled continuously."""

    def __init__(self, per_minute: float):
        self.capacity = per_minute
        self.tokens = per_minute
        self.rate = per_minute / 60.0
        self.last = time.monotonic()

    def take(self) -> None:
        while True:
            now = time.monotonic()
            self.tokens = min(self.capacity, self.tokens + (now - self.last) * self.rate)
            self.last = now
            if self.tokens >= 1:
                self.tokens -= 1
                return
            time.sleep((1 - self.tokens) / self.rate)


class SuperMarketClient:
    def __init__(self, api_key: Optional[str] = None, base_url: Optional[str] = None,
                 reads_per_min: float = 90, writes_per_min: float = 27, timeout: float = 20):
        self.key = api_key or os.environ.get("SUPER_MARKET_API_KEY")
        if not self.key:
            raise RuntimeError("Set SUPER_MARKET_API_KEY (My Profile -> API Keys; scopes: read, trade)")
        self.base = (base_url or os.environ.get("SUPER_MARKET_BASE_URL") or DEFAULT_BASE).rstrip("/")
        self.reads = TokenBucket(reads_per_min)     # spec: 100/min; keep headroom
        self.writes = TokenBucket(writes_per_min)   # spec: 30/min
        self.timeout = timeout

    # ---- transport ------------------------------------------------------

    def request(self, method: str, path: str, params: Optional[Dict[str, Any]] = None,
                body: Optional[Dict[str, Any]] = None, max_attempts: int = 5) -> Any:
        params = {k: v for k, v in (params or {}).items() if v is not None}
        url = self.base + path + ("?" + urllib.parse.urlencode(params) if params else "")
        data = json.dumps(body).encode() if body is not None else None
        bucket = self.reads if method in ("GET", "HEAD") else self.writes
        delay = 0.5
        for attempt in range(1, max_attempts + 1):
            bucket.take()
            req = urllib.request.Request(url, data=data, method=method, headers={
                "Authorization": f"Bearer {self.key}",
                "Content-Type": "application/json",
                "Accept": "application/json",
            })
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read()
                    return json.loads(raw) if raw else None
            except urllib.error.HTTPError as e:
                raw = e.read()
                try:
                    err = json.loads(raw).get("error", {})
                except (ValueError, AttributeError):
                    err = {}
                code = err.get("code", "HTTP_ERROR")
                if e.code == 207:  # batch with partial success is a valid result
                    return json.loads(raw)
                # 502 covers ORDER_STATUS_UNKNOWN, including batch bodies that
                # carry per-item results and no top-level code.
                retryable = (e.code in (429, 502) or code in ("TX_CONFLICT", "SERVICE_UNAVAILABLE",
                                                       "ORDER_STATUS_UNKNOWN", "REQUEST_IN_FLIGHT"))
                if not retryable or attempt == max_attempts:
                    raise ApiError(e.code, code, err.get("message", raw[:200]), err.get("details"))
                retry_after = e.headers.get("Retry-After")
                if code == "REQUEST_IN_FLIGHT":
                    wait = 90.0  # spec: the idempotency lease lasts 90 seconds
                elif retry_after:
                    wait = float(retry_after)
                else:
                    wait = delay + random.uniform(0, delay)
                    delay *= 2
                time.sleep(wait)
            except (urllib.error.URLError, TimeoutError):
                if attempt == max_attempts:
                    raise
                time.sleep(delay + random.uniform(0, delay))
                delay *= 2

    def get(self, path: str, **params) -> Any:
        return self.request("GET", path, params)

    def post(self, path: str, body: Dict[str, Any]) -> Any:
        return self.request("POST", path, body=body)

    def paginate(self, path: str, key: str = "data", max_pages: int = 50, **params) -> Iterator[dict]:
        cursor = None
        for _ in range(max_pages):
            page = self.get(path, cursor=cursor, **params)
            yield from page.get(key, [])
            pg = page.get("pagination") or {}
            cursor = pg.get("nextCursor")
            if not pg.get("hasMore") or not cursor:
                return

    # ---- endpoints used by the bot ----------------------------------------

    def tournament(self, slug: str) -> dict:
        return self.get(f"/tournaments/{slug}")

    def tournaments(self) -> List[dict]:
        r = self.get("/tournaments")
        return r.get("data", r) if isinstance(r, dict) else r

    def markets(self, tournament_id: Optional[str], status: str = "open") -> List[dict]:
        return list(self.paginate("/markets", limit=100, status=status, tournamentId=tournament_id))

    def market_orderbook(self, market_id: str, tournament_id: Optional[str], depth: int = 10) -> dict:
        return self.get(f"/markets/{market_id}/orderbook", tournamentId=tournament_id, depth=depth)

    def prices(self, exchange_ids: List[str], tournament_id: Optional[str]) -> List[dict]:
        out = []
        for i in range(0, len(exchange_ids), 100):  # endpoint takes at most 100 ids
            r = self.get("/exchanges/prices", ids=",".join(exchange_ids[i:i + 100]), tournamentId=tournament_id)
            out.extend(r.get("data", []))
        return out

    def relationships(self, tournament_id: Optional[str], rel_type: Optional[str] = None,
                      market_id: Optional[str] = None) -> List[dict]:
        return list(self.paginate("/relationships", limit=100, type=rel_type,
                                  tournamentId=tournament_id, marketId=market_id))

    def price_history(self, exchange_id: str, tournament_id: Optional[str], resolution: str = "1h",
                      start: Optional[str] = None, limit: int = 1000) -> dict:
        return self.get(f"/exchanges/{exchange_id}/price-history", tournamentId=tournament_id,
                        resolution=resolution, limit=limit, **({"from": start} if start else {}))

    def positions(self, slug: str) -> dict:
        return self.get(f"/tournaments/{slug}/portfolio/positions")

    def cancel_all(self, tournament_id: Optional[str]) -> Any:
        return self.post("/orders/cancel-all", {"tournamentId": tournament_id} if tournament_id else {})

    def place_batch(self, orders: List[dict]) -> Any:
        """Best-effort batch of up to 50 orders. One write against the rate limit."""
        return self.post("/orders/batch", {"idempotencyKey": str(uuid.uuid4()), "orders": orders})

    def place_multi_leg(self, legs: List[dict], relationship_id: Optional[str] = None) -> Any:
        """Atomic: every leg is placed or none is. Used for arbitrage baskets."""
        body: Dict[str, Any] = {"idempotencyKey": str(uuid.uuid4()), "legs": legs}
        if relationship_id:
            body["relationshipConstraint"] = relationship_id
        return self.post("/orders/multi-leg", body)

# =============================================================================
# Live trading loop
# =============================================================================

def _hours_until(ts: Optional[str]) -> float:
    if not ts:
        return float("inf")
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return float("inf")
    return (dt - datetime.now(timezone.utc)).total_seconds() / 3600


def _levels(side: List[dict]) -> List[Tuple[float, float]]:
    return [(float(l["price"]), float(l["quantity"])) for l in side or [] if float(l["quantity"]) > 0]


class Bot:
    def __init__(self, client: SuperMarketClient, slug: str, params: StrategyParams,
                 live: bool, max_books: int, max_drawdown: float):
        self.c = client
        self.slug = slug
        self.strategy = Strategy(params)
        self.params = params
        self.live = live
        self.max_books = max_books
        self.max_drawdown = max_drawdown
        self.tournament_id: Optional[str] = None
        self.markets: Dict[str, dict] = {}            # market id -> market row
        self.baskets: Dict[str, Tuple[bool, str]] = {}  # market id -> (exhaustive, relationship id)
        self.universe_at = 0.0
        self.peak_equity = 0.0

    # ---- universe ---------------------------------------------------------

    def refresh_universe(self) -> None:
        """Open markets and the mutually-exclusive relationships between their
        outcomes. Refreshed every 15 minutes, not every cycle."""
        t = self.c.tournament(self.slug)
        self.tournament_id = t["id"]
        self.markets = {m["id"]: m for m in self.c.markets(self.tournament_id, status="open")}
        self.baskets = {}
        ex_to_market = {e["id"]: mid for mid, m in self.markets.items() for e in m.get("exchanges", [])}
        try:
            rels = self.c.relationships(self.tournament_id, rel_type="mutually_exclusive")
        except ApiError as e:
            print(f"[warn] relationships unavailable ({e}); basket arbitrage disabled this period")
            rels = []
        for r in rels:
            if r.get("status") != "active":
                continue
            ids = {n["exchangeId"] for n in r.get("nodes", [])}
            mids = {ex_to_market.get(i) for i in ids}
            # Only baskets that cover exactly one whole market are traded as
            # baskets. Cross-market relationships are left alone for safety.
            if len(mids) == 1 and None not in mids:
                mid = mids.pop()
                if ids == {e["id"] for e in self.markets[mid]["exchanges"]}:
                    self.baskets[mid] = (bool(r.get("isExhaustive")), r["id"])
        self.universe_at = time.time()
        print(f"[universe] {len(self.markets)} open markets, {len(self.baskets)} exclusive baskets")

    # ---- portfolio ----------------------------------------------------------

    def portfolio(self) -> Portfolio:
        cash = float(self.c.tournament(self.slug)["myBalance"])
        data = self.c.positions(self.slug)
        positions: Dict[str, Position] = {}
        value = 0.0
        for p in data.get("positions", []):
            if p.get("settled"):
                continue
            q = float(p["quantity"])        # spec: positive = YES, negative = NO
            cost = float(p.get("costBasis", 0.0))
            positions[p["exchangeId"]] = (Position(yes_qty=q, yes_cost=cost) if q > 0
                                          else Position(no_qty=-q, no_cost=cost))
            value += float(p.get("marketValue", 0.0))
        return Portfolio(cash=cash, equity=cash + value, positions=positions)

    # ---- screening ----------------------------------------------------------

    def screen(self, pf: Portfolio) -> List[str]:
        """Score markets from the cheap bulk snapshot; return the ids worth a full book read."""
        all_ex = [e["id"] for m in self.markets.values() for e in m.get("exchanges", [])]
        quotes = {q["exchangeId"]: q for q in self.c.prices(all_ex, self.tournament_id)}
        for ex_id, q in quotes.items():
            self.strategy.observe_price(ex_id, q.get("latestPrice"))
        scores: Dict[str, float] = {}
        for mid, m in self.markets.items():
            qs = [quotes.get(e["id"], {}) for e in m.get("exchanges", [])]
            if _hours_until(m.get("settlementDate")) <= 0:
                continue
            score = 0.0
            for q in qs:
                if q.get("spread") is not None:
                    score += max(0.0, q["spread"] - self.params.mm_min_book_spread)
            if mid in self.baskets and all(q.get("bestAsk") is not None for q in qs):
                if sum(q["bestAsk"] for q in qs) < 1:          # YES-basket arbitrage visible at the touch
                    score += 10
            if mid in self.baskets and all(q.get("bestBid") is not None for q in qs):
                if sum(q["bestBid"] for q in qs) > 1:          # NO-basket arbitrage
                    score += 10
            if any(e["id"] in pf.positions for e in m.get("exchanges", [])):
                score += 1                                     # always manage open risk
            if score > 0:
                scores[mid] = score
        return sorted(scores, key=scores.get, reverse=True)[: self.max_books]

    def build_groups(self, market_ids: List[str]) -> List[GroupView]:
        groups = []
        for mid in market_ids:
            m = self.markets[mid]
            book = self.c.market_orderbook(mid, self.tournament_id)
            exs = book.get("exchanges", [])
            for ctx in book.get("contexts", []) or []:
                if (ctx.get("tournament") or {}).get("id") == self.tournament_id and ctx.get("exchanges"):
                    exs = ctx["exchanges"]
            views = [ExchangeView(e["exchangeId"], _levels(e.get("bids")), _levels(e.get("asks")))
                     for e in exs]
            hrs = _hours_until(m.get("settlementDate"))
            if mid in self.baskets:
                exhaustive, _ = self.baskets[mid]
                groups.append(GroupView(mid, mid, views, exclusive=True, exhaustive=exhaustive,
                                        hours_to_settle=hrs))
            else:
                # Not known to be mutually exclusive: treat each outcome as its own binary.
                for v in views:
                    groups.append(GroupView(f"{mid}:{v.exchange_id}", mid, [v], hours_to_settle=hrs))
        return groups

    # ---- execution ----------------------------------------------------------

    def _order(self, it: OrderIntent) -> dict:
        return {"exchangeId": it.exchange_id, "side": it.side, "action": "buy",
                "quantity": int(it.qty), "price": round(it.price, 3), "tournamentId": self.tournament_id}

    def execute(self, intents: List[OrderIntent]) -> None:
        for it in intents:
            print(f"  {it.kind:<5} {it.exchange_id:>8} buy {it.side.upper():<3} {it.qty:>6} @ {it.price:.3f}")
        if not self.live:
            return
        self.c.cancel_all(self.tournament_id)
        baskets: Dict[str, List[OrderIntent]] = {}
        singles: List[OrderIntent] = []
        for it in intents:
            (baskets.setdefault(it.basket_id, []) if it.basket_id else singles).append(it)
        # Arbitrage first, atomically: either every leg is placed or none is.
        # Do NOT pass relationshipConstraint here. The engine uses it to demand
        # that leg prices sum to 1 (exhaustive) or at most 1, and an arbitrage
        # basket violates that by construction, so every arb would be rejected.
        # Legs are limit orders, so a leg can still rest partly unfilled if the
        # book moved since we read it. Next cycle's cancel-all clears the rest,
        # and the risk limits treat any leftover as an ordinary position.
        for bid, legs in baskets.items():
            try:
                self.c.place_multi_leg([self._order(l) for l in legs])
            except ApiError as e:
                print(f"[arb rejected] {bid}: {e}")
        for i in range(0, len(singles), 50):
            try:
                r = self.c.place_batch([self._order(o) for o in singles[i:i + 50]])
                failed = [x for x in (r or {}).get("results", []) if not x.get("ok", True)]
                if failed:
                    print(f"[batch] {len(failed)} orders rejected, first: {json.dumps(failed[0])[:200]}")
            except ApiError as e:
                print(f"[batch error] {e}")

    # ---- main loop ----------------------------------------------------------

    def run(self, cycle_seconds: float, max_cycles: Optional[int]) -> None:
        n = 0
        while max_cycles is None or n < max_cycles:
            n += 1
            start = time.time()
            try:
                if time.time() - self.universe_at > 900 or not self.markets:
                    self.refresh_universe()
                pf = self.portfolio()
                self.peak_equity = max(self.peak_equity, pf.equity)
                if self.peak_equity and pf.equity < (1 - self.max_drawdown) * self.peak_equity:
                    print(f"[circuit breaker] equity {pf.equity:.2f} is more than "
                          f"{self.max_drawdown:.0%} below peak {self.peak_equity:.2f}; cancelling and stopping")
                    if self.live:
                        self.c.cancel_all(self.tournament_id)
                    return
                groups = self.build_groups(self.screen(pf))
                intents = self.strategy.decide(groups, pf)
                print(f"[cycle {n}] cash {pf.cash:.2f} equity {pf.equity:.2f} "
                      f"positions {len(pf.positions)} groups {len(groups)} orders {len(intents)}"
                      f"{'' if self.live else ' (dry run)'}")
                self.execute(intents)
            except ApiError as e:
                print(f"[api error] {e}")
            except Exception as e:  # keep the loop alive; the next cycle re-syncs from REST
                print(f"[error] {type(e).__name__}: {e}")
            time.sleep(max(0.0, cycle_seconds - (time.time() - start)))

# =============================================================================
# Entry point
# =============================================================================

def main() -> None:
    if "PASTE" in API_KEY:
        print("Open this file and paste your API key into API_KEY at the top.")
        return
    client = SuperMarketClient(API_KEY, BASE_URL)
    if "PASTE" in TOURNAMENT_SLUG:
        print("Tournaments you can trade in (copy a slug into TOURNAMENT_SLUG):")
        for t in client.tournaments():
            print(f"  slug: {t['slug']:<40} name: {t['name']}  balance: {t.get('myBalance')}")
        return
    params = replace(StrategyParams(), longshot_gamma=LONGSHOT_GAMMA)
    print(f"Starting bot on '{TOURNAMENT_SLUG}' - "
          f"{'LIVE TRADING' if LIVE_TRADING else 'DRY RUN (no orders placed)'}. Ctrl+C to stop.")
    bot = Bot(client, TOURNAMENT_SLUG, params, LIVE_TRADING, MAX_BOOKS_PER_CYCLE, MAX_DRAWDOWN)
    try:
        bot.run(CYCLE_SECONDS, None)
    except KeyboardInterrupt:
        print("\nStopped. Resting orders stay on the book until cancelled.")


if __name__ == "__main__":
    main()
