"""Profit-maximising strategy for the SIG Predictions Cup (Super Market API).

Pure decision logic with no network calls. The live bot (``bot.py``) and the
backtester (``backtest.py``) both feed it the same view of the market, so what
gets backtested is exactly what trades.

How a contract pays
-------------------
Every exchange is a binary contract. YES pays 1 if the outcome happens and 0
otherwise; NO pays the reverse. All prices are YES-normalised in [0, 1], so a
YES bid at b is the same order as a NO offer at 1 - b. There are no trading
fees, so any edge we capture is kept, and holding a contract until settlement
costs nothing.

The strategy has three profit engines, run in priority order:

1. Arbitrage (risk-free). In a mutually exclusive basket at most one outcome
   can win. If the best YES asks sum below 1 (and the basket is exhaustive),
   buying one of each costs under 1 and pays exactly 1. If the best YES bids
   sum above 1, buying one NO of each costs sum(1 - bid) < n - 1 and pays at
   least n - 1. The profit is locked in at trade time.

2. Taking (positive expected value). Estimate a fair probability q for each
   contract (book mid + trade EWMA, normalised across the basket, then
   corrected for favourite-longshot bias). When an ask sits below q, or a bid
   above q, by more than ``min_take_edge``, cross the spread. Size with
   fractional Kelly, capped by risk limits.

3. Market making (spread capture). Where the book is wide, rest a bid below
   fair value and an ask above it. Uninformed flow that hits us pays the
   spread. Quotes are skewed against existing inventory so the book pulls our
   position back toward flat. Quoting stops near settlement, when informed
   traders and late news make resting orders a liability.

Risk limits bound the worst case: a maximum loss per contract and per market,
plus a cash reserve that is never deployed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import MAX_PRICE, MIN_PRICE, TICK, StrategyParams

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
