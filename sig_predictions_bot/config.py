"""Tunable parameters for the strategy, shared by the live bot and the backtester.

Every number here is a knob the backtest can sweep. The defaults are the
values the synthetic backtest favoured. Re-check them against real settled
history (see ``history.py`` / ``backtest.py --real``) before trusting them.
"""

from dataclasses import dataclass

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
