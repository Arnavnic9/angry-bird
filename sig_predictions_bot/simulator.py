"""Synthetic prediction-market world used to backtest the strategy.

Why simulate? The sandbox this was built in cannot reach the Super Market API,
and an API key is needed to download real history. ``backtest.py --real``
replays real settled markets once you have downloaded them with
``history.py``. This simulator is for testing logic, risk limits and the
relative value of each component under controlled conditions.

How the world works
-------------------
* Truth. Each market secretly picks its winner at creation (from a random
  prior). Each hour, every outcome emits a noisy signal that leans slightly
  toward the true winner. The true probability is the exact Bayesian
  posterior given those signals, so it is a martingale and perfectly
  calibrated, the way real probabilities behave. Occasional "news" hours
  carry a much stronger signal (sudden price jumps).

* The crowd. The visible consensus price differs from the truth in three
  realistic ways:
    1. lag: it closes only part of the gap to the truth each hour,
    2. noise: an autocorrelated error (sentiment that drifts, then reverts),
    3. favourite-longshot bias: logit(crowd) = logit(truth) / gamma_true,
       which pulls extreme prices toward 50%.
  In multi-outcome markets each outcome's noise is independent, so the
  prices do not sum to exactly 1. Competing arbitrageurs remove most of that
  gap (``crowd_coherence``). The remainder is what basket arbitrage can earn.

* The order book. Background market makers quote a ladder around the crowd
  price with a market-specific spread.

* Order flow. Each hour a random number of takers arrive. Most are noise
  traders who pick a side at random. Some are informed: they know the true
  probability and only trade when the best price is wrong. Informed flow is
  what makes market making risky (adverse selection), so leaving it out would
  flatter the strategy.

Our fills are modelled conservatively. At an equal price we queue behind the
background liquidity that was there first, and we get filled only by size
left over after it.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .config import MAX_PRICE, MIN_PRICE, TICK


def sigmoid(x: float) -> float:
    if x >= 0:
        return 1 / (1 + math.exp(-x))
    e = math.exp(x)
    return e / (1 + e)


def logit(p: float) -> float:
    p = min(max(p, 1e-6), 1 - 1e-6)
    return math.log(p / (1 - p))


def biased(probs: List[float], gamma: float) -> List[float]:
    """Favourite-longshot bias: the crowd prices outcome k at p_k^(1/gamma),
    renormalised. For a binary this is logit(price) = logit(p) / gamma.
    It is the exact inverse of the correction the strategy applies."""
    w = [max(p, 1e-9) ** (1 / gamma) for p in probs]
    z = sum(w)
    return [x / z for x in w]


@dataclass
class WorldParams:
    hours: int = 24 * 60                 # length of the backtest (60 days, hourly steps)
    market_arrival_per_day: float = 1.5  # new markets listed per day
    initial_markets: int = 25
    min_life_h: int = 48
    max_life_h: int = 24 * 21
    multi_outcome_share: float = 0.35
    min_outcomes: int = 3
    max_outcomes: int = 6
    info_total: float = 9.0              # total signal (sum of mu^2) over a market's life
    news_prob: float = 0.015             # chance an hour carries big news
    news_mult: float = 4.0
    gamma_true: float = 1.15             # crowd favourite-longshot bias (1 = none)
    crowd_adjust: float = 0.5            # share of the gap to the truth the crowd closes per hour
    crowd_noise_sd: float = 0.12         # innovation sd of the crowd's logit error
    crowd_noise_ar: float = 0.92         # persistence of the crowd error
    crowd_coherence: float = 0.85        # share of a multi-outcome basket's mispricing other arbitrageurs remove
    min_half_spread_ticks: int = 1
    max_half_spread_ticks: int = 6
    depth_min: float = 20
    depth_max: float = 250
    levels: int = 6
    takers_per_hour_min: float = 0.3
    takers_per_hour_max: float = 3.0
    informed_share: float = 0.25
    taker_size_min: int = 5
    taker_size_max: int = 80


@dataclass
class SimMarket:
    market_id: str
    start: int
    end: int
    n_outcomes: int          # 2 for a binary (only outcome 0 is listed as an exchange)
    winner: int
    mu: float
    log_post: List[float]
    crowd_logit: List[float]
    crowd_err: List[float]
    half_spread_ticks: int
    depth: float
    activity: float
    books: Dict[str, Tuple[list, list]] = field(default_factory=dict)
    last_trade: Dict[str, Optional[float]] = field(default_factory=dict)
    hour_trades: Dict[str, List[Tuple[float, int]]] = field(default_factory=dict)  # (price, size) this hour

    @property
    def is_binary(self) -> bool:
        return self.n_outcomes == 2

    @property
    def exchange_ids(self) -> List[str]:
        if self.is_binary:
            return [f"{self.market_id}-0"]
        return [f"{self.market_id}-{k}" for k in range(self.n_outcomes)]

    def outcome_index(self, exchange_id: str) -> int:
        return int(exchange_id.rsplit("-", 1)[1])

    def true_probs(self) -> List[float]:
        m = max(self.log_post)
        w = [math.exp(v - m) for v in self.log_post]
        z = sum(w)
        return [x / z for x in w]

    coherence: float = 0.85

    def crowd_price(self, k: int) -> float:
        c = sigmoid(self.crowd_logit[k])
        if self.is_binary:
            return c
        # Other traders arbitrage the basket. They remove most, but not all,
        # of the gap between the sum of prices and 1.
        s = sum(sigmoid(x) for x in self.crowd_logit)
        lam = self.coherence
        return c * (1 - lam + lam / s)

    def pays_yes(self, exchange_id: str) -> bool:
        return self.outcome_index(exchange_id) == self.winner


class World:
    """Steps every live market forward one hour at a time."""

    def __init__(self, params: WorldParams, seed: int):
        self.wp = params
        self.rng = random.Random(seed)
        self.t = 0
        self.markets: Dict[str, SimMarket] = {}
        self.settled: List[SimMarket] = []
        self._next_id = 1
        for _ in range(params.initial_markets):
            self._new_market(start=0)

    # ---- market lifecycle ------------------------------------------------

    def _new_market(self, start: int) -> None:
        wp, r = self.wp, self.rng
        mid = f"m{self._next_id}"
        self._next_id += 1
        life = r.randint(wp.min_life_h, wp.max_life_h)
        if r.random() < wp.multi_outcome_share:
            n = r.randint(wp.min_outcomes, wp.max_outcomes)
            # Dirichlet-ish prior: a few favourites, a tail of longshots.
            raw = [r.gammavariate(0.8, 1.0) for _ in range(n)]
        else:
            n = 2
            p_yes = min(0.97, max(0.03, r.betavariate(1.2, 1.2)))
            raw = [p_yes, 1 - p_yes]
        z = sum(raw)
        prior = [max(x / z, 1e-3) for x in raw]
        z = sum(prior)
        prior = [x / z for x in prior]
        winner = r.choices(range(n), weights=prior)[0]
        mu = math.sqrt(wp.info_total / life)
        log_post = [math.log(p) for p in prior]
        crowd_logit = [logit(q) for q in biased(prior, wp.gamma_true)]
        m = SimMarket(
            market_id=mid, start=start, end=start + life, n_outcomes=n, winner=winner,
            mu=mu, log_post=log_post, crowd_logit=crowd_logit, crowd_err=[0.0] * n,
            half_spread_ticks=r.randint(wp.min_half_spread_ticks, wp.max_half_spread_ticks),
            depth=r.uniform(wp.depth_min, wp.depth_max),
            activity=r.uniform(wp.takers_per_hour_min, wp.takers_per_hour_max),
            coherence=wp.crowd_coherence,
        )
        for e in m.exchange_ids:
            m.last_trade[e] = None
        self.markets[mid] = m
        self._build_books(m)

    # ---- hourly update ---------------------------------------------------

    def advance(self) -> None:
        """Move to the next hour: new information, crowd repricing, fresh books."""
        wp, r = self.wp, self.rng
        self.t += 1
        # New listings arrive as a Poisson process.
        lam = wp.market_arrival_per_day / 24
        while r.random() < lam:
            self._new_market(start=self.t)
            lam *= 0.5
        for m in self.markets.values():
            m.hour_trades = {e: [] for e in m.exchange_ids}
            # Truth: Bayesian update on this hour's signals.
            mu = m.mu * (wp.news_mult if r.random() < wp.news_prob else 1.0)
            for k in range(m.n_outcomes):
                y = (mu if k == m.winner else 0.0) + r.gauss(0, 1)
                m.log_post[k] += mu * y
            biased_probs = biased(m.true_probs(), wp.gamma_true)
            # Crowd: partial adjustment toward a biased, noisy view of the truth.
            for k in range(m.n_outcomes):
                m.crowd_err[k] = wp.crowd_noise_ar * m.crowd_err[k] + r.gauss(0, wp.crowd_noise_sd)
                target = logit(biased_probs[k]) + m.crowd_err[k]
                m.crowd_logit[k] += wp.crowd_adjust * (target - m.crowd_logit[k])
            self._build_books(m)

    def _build_books(self, m: SimMarket) -> None:
        """Background makers quote a ladder around the crowd price."""
        wp, r = self.wp, self.rng
        for e in m.exchange_ids:
            c = m.crowd_price(m.outcome_index(e))
            h = max(1, m.half_spread_ticks + r.randint(-1, 1)) * TICK
            bb = math.floor((c - h) / TICK) * TICK
            ba = math.ceil((c + h) / TICK) * TICK
            if ba - bb < TICK - 1e-9:
                ba = bb + TICK
            bids, asks = [], []
            px = bb
            for _ in range(wp.levels):
                if px >= MIN_PRICE - 1e-9:
                    bids.append([round(px, 3), round(m.depth * r.uniform(0.5, 1.5))])
                px -= TICK * r.randint(1, 2)
            px = ba
            for _ in range(wp.levels):
                if px <= MAX_PRICE + 1e-9:
                    asks.append([round(px, 3), round(m.depth * r.uniform(0.5, 1.5))])
                px += TICK * r.randint(1, 2)
            m.books[e] = (bids, asks)

    def settle_due(self) -> List[SimMarket]:
        """Remove and return markets whose settlement time has arrived."""
        due = [m for m in self.markets.values() if m.end <= self.t]
        for m in due:
            del self.markets[m.market_id]
            self.settled.append(m)
        return due

    # ---- order flow ------------------------------------------------------

    def taker_flow(self, m: SimMarket, exchange_id: str, our_bid: Optional[Tuple[float, int]],
                   our_ask: Optional[Tuple[float, int]]) -> Tuple[int, int]:
        """Send this hour's takers into the book, which holds the background
        ladder plus our resting quotes.

        ``our_bid`` is (YES price, qty) and ``our_ask`` is (YES price, qty).
        Returns (qty of our bid filled, qty of our ask filled).
        """
        wp, r = self.wp, self.rng
        bids, asks = m.books[exchange_id]
        k = m.outcome_index(exchange_id)
        true_p = m.true_probs()[k]
        n_takers = self._poisson(m.activity)
        bid_left = our_bid[1] if our_bid else 0
        ask_left = our_ask[1] if our_ask else 0
        filled_bid = filled_ask = 0
        bg_bid = [list(x) for x in bids]
        bg_ask = [list(x) for x in asks]

        for _ in range(n_takers):
            size = r.randint(wp.taker_size_min, wp.taker_size_max)
            best_ask = min([a for a, _ in bg_ask[:1]] + ([our_ask[0]] if ask_left > 0 else []), default=None)
            best_bid = max([b for b, _ in bg_bid[:1]] + ([our_bid[0]] if bid_left > 0 else []), default=None)
            if r.random() < wp.informed_share:
                # Informed taker: trades only when the touch is mispriced against the truth.
                if best_ask is not None and true_p > best_ask + TICK:
                    buy_yes = True
                elif best_bid is not None and true_p < best_bid - TICK:
                    buy_yes = False
                else:
                    continue
            else:
                buy_yes = r.random() < 0.5

            if buy_yes:
                # Sweeps asks from the lowest. Our ask fills first if strictly better,
                # otherwise only after background size at the same price.
                f, last = self._sweep(size, bg_ask, our_ask[0] if ask_left > 0 else None, ask_left, ascending=True)
                filled_ask += f
                ask_left -= f
            else:
                f, last = self._sweep(size, bg_bid, our_bid[0] if bid_left > 0 else None, bid_left, ascending=False)
                filled_bid += f
                bid_left -= f
            if last is not None:
                m.last_trade[exchange_id] = last
                m.hour_trades.setdefault(exchange_id, []).append((last, size))
        return filled_bid, filled_ask

    @staticmethod
    def _sweep(size, levels, our_px, our_qty, ascending):
        """Consume ``size`` from a price ladder that includes our order.
        Returns (our quantity filled, last traded price)."""
        filled_ours, last = 0, None
        remaining = size
        while remaining > 0:
            bg_px = levels[0][0] if levels else None
            ours_live = our_px is not None and our_qty - filled_ours > 0
            if bg_px is None and not ours_live:
                break
            better = (lambda a, b: a < b - 1e-9) if ascending else (lambda a, b: a > b + 1e-9)
            if ours_live and (bg_px is None or better(our_px, bg_px)):
                take = min(remaining, our_qty - filled_ours)
                filled_ours += take
                remaining -= take
                last = our_px
                continue
            # Background level goes first, including at equal price (queue priority).
            take = min(remaining, levels[0][1])
            levels[0][1] -= take
            remaining -= take
            last = bg_px
            if levels[0][1] <= 0:
                levels.pop(0)
        return filled_ours, last

    def _poisson(self, lam: float) -> int:
        # Knuth's method. Fine for the small rates used here.
        L, k, p = math.exp(-lam), 0, 1.0
        while True:
            p *= self.rng.random()
            if p <= L:
                return k
            k += 1
