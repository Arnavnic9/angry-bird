"""Backtester for the SIG Predictions Cup strategy.

Two modes:

  Synthetic (default). Runs the strategy in the simulated world from
  ``simulator.py`` over many random seeds. It reports the return
  distribution, drawdown, PnL from each component (arb / take / quote),
  ablations with one component on at a time, naive baselines, and a stress
  test in a world with no favourite-longshot bias.

      python -m sig_predictions_bot.backtest --seeds 20

  Real-data replay. Replays settled Super Market markets downloaded with
  ``history.py download``. Candles stand in for the book: we assume a
  half-spread around the latest close, and a resting quote fills only if the
  next candle trades through it.

      python -m sig_predictions_bot.backtest --real history.json

Accounting. Cash moves on every fill. A YES/NO pair in the same exchange is
netted into 1 unit of cash at once, as the engine does. At settlement each
remaining contract pays 1 or 0. Every fill is credited
qty * (payout - price), so the PnL attribution by component sums exactly to
the change in equity.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional, Tuple

from .config import StrategyParams, TICK
from .simulator import World, WorldParams
from .strategy import ExchangeView, GroupView, OrderIntent, Portfolio, Position, Strategy


# ----------------------------------------------------------------------------
# Account
# ----------------------------------------------------------------------------

@dataclass
class Fill:
    t: int
    exchange_id: str
    side: str
    price: float
    qty: int
    kind: str


class Account:
    def __init__(self, cash: float):
        self.initial = cash
        self.cash = cash
        self.positions: Dict[str, Position] = {}
        self.fills: List[Fill] = []
        self.pnl_by_kind: Dict[str, float] = {"arb": 0.0, "take": 0.0, "quote": 0.0, "other": 0.0}
        self._open_fills: Dict[str, List[Fill]] = {}

    def buy(self, t: int, ex: str, side: str, price: float, qty: int, kind: str) -> int:
        """Execute a buy. Cash can never go negative, as with tournament balances."""
        qty = int(min(qty, self.cash / price)) if price > 0 else qty
        if qty <= 0:
            return 0
        self.cash -= qty * price
        ps = self.positions.setdefault(ex, Position())
        # Net against the opposite side: each YES+NO pair is worth exactly 1.
        if side == "yes":
            pair = min(qty, ps.no_qty)
            if pair:
                ps.no_cost -= ps.no_cost * pair / ps.no_qty
                ps.no_qty -= pair
            ps.yes_qty += qty - pair
            ps.yes_cost += (qty - pair) * price
        else:
            pair = min(qty, ps.yes_qty)
            if pair:
                ps.yes_cost -= ps.yes_cost * pair / ps.yes_qty
                ps.yes_qty -= pair
            ps.no_qty += qty - pair
            ps.no_cost += (qty - pair) * price
        self.cash += pair
        f = Fill(t, ex, side, price, qty, kind)
        self.fills.append(f)
        self._open_fills.setdefault(ex, []).append(f)
        return qty

    def settle(self, ex: str, yes_wins: bool) -> None:
        ps = self.positions.pop(ex, None)
        if ps is not None:
            self.cash += ps.yes_qty if yes_wins else ps.no_qty
        for f in self._open_fills.pop(ex, []):
            payout = 1.0 if (f.side == "yes") == yes_wins else 0.0
            key = f.kind if f.kind in self.pnl_by_kind else "other"
            self.pnl_by_kind[key] += f.qty * (payout - f.price)

    def equity(self, marks: Dict[str, float]) -> float:
        """Cash plus open positions marked at the given YES prices."""
        v = self.cash
        for ex, ps in self.positions.items():
            m = marks.get(ex)
            if m is None:
                v += ps.yes_cost + ps.no_cost  # unpriced: carry at cost
            else:
                v += ps.yes_qty * m + ps.no_qty * (1 - m)
        return v

    def portfolio(self, marks: Dict[str, float]) -> Portfolio:
        return Portfolio(cash=self.cash, equity=self.equity(marks), positions=self.positions)


# ----------------------------------------------------------------------------
# Execution helpers (shared by both modes)
# ----------------------------------------------------------------------------

def execute_against_book(acct: Account, t: int, intent: OrderIntent, bids: list, asks: list) -> int:
    """Fill a marketable order against a visible book, level by level, at each
    level's own price. Any unfilled remainder is cancelled (the live bot cancels
    leftovers at the start of the next cycle). Mutates the book."""
    filled = 0
    remaining = intent.qty
    if intent.side == "yes":
        while remaining > 0 and asks and asks[0][0] <= intent.price + 1e-9:
            px, sz = asks[0]
            q = acct.buy(t, intent.exchange_id, "yes", px, int(min(sz, remaining)), intent.kind)
            if q <= 0:
                break
            filled += q
            remaining -= q
            asks[0][1] -= q
            if asks[0][1] <= 0:
                asks.pop(0)
    else:
        floor_yes = 1 - intent.price
        while remaining > 0 and bids and bids[0][0] >= floor_yes - 1e-9:
            px, sz = bids[0]
            q = acct.buy(t, intent.exchange_id, "no", round(1 - px, 3), int(min(sz, remaining)), intent.kind)
            if q <= 0:
                break
            filled += q
            remaining -= q
            bids[0][1] -= q
            if bids[0][1] <= 0:
                bids.pop(0)
    return filled


@dataclass
class RunResult:
    final_equity: float
    initial: float
    equity_curve: List[float]
    pnl_by_kind: Dict[str, float]
    n_fills: int
    markets_traded: int
    markets_won: int

    @property
    def ret(self) -> float:
        return self.final_equity / self.initial - 1

    @property
    def max_drawdown(self) -> float:
        peak, dd = -math.inf, 0.0
        for v in self.equity_curve:
            peak = max(peak, v)
            dd = max(dd, (peak - v) / peak if peak > 0 else 0)
        return dd

    @property
    def daily_sharpe(self) -> float:
        """Annualised Sharpe ratio of daily equity changes (a smoothness gauge,
        not a promise)."""
        daily = self.equity_curve[::24]
        rets = [b / a - 1 for a, b in zip(daily, daily[1:]) if a > 0]
        if len(rets) < 2 or statistics.pstdev(rets) == 0:
            return 0.0
        return statistics.mean(rets) / statistics.pstdev(rets) * math.sqrt(365)


# ----------------------------------------------------------------------------
# Synthetic backtest
# ----------------------------------------------------------------------------

class RandomTrader:
    """Baseline: every so often, buy a random side at the touch. It shows what
    paying the spread with no edge costs."""

    def __init__(self, seed: int, rate: float = 0.03, frac: float = 0.01):
        self.rng = random.Random(seed)
        self.rate, self.frac = rate, frac

    def observe_price(self, *_):
        pass

    def decide(self, groups: List[GroupView], pf: Portfolio) -> List[OrderIntent]:
        out = []
        for g in groups:
            for ex in g.exchanges:
                if self.rng.random() > self.rate or not ex.bids or not ex.asks:
                    continue
                if self.rng.random() < 0.5:
                    px = ex.asks[0][0]
                    out.append(OrderIntent(ex.exchange_id, "yes", px, int(self.frac * pf.equity / px), "take"))
                else:
                    px = round(1 - ex.bids[0][0], 3)
                    out.append(OrderIntent(ex.exchange_id, "no", px, int(self.frac * pf.equity / px), "take"))
        return [o for o in out if o.qty > 0]


def run_synthetic(make_strategy: Callable[[], object], wp: WorldParams, seed: int,
                  initial_cash: float = 10_000.0, stale_quotes: bool = False) -> RunResult:
    """Run one synthetic world.

    ``stale_quotes=False``: quotes are set against the current book and rest
    through that hour's order flow. This stands for a bot that re-quotes faster
    than prices move.
    ``stale_quotes=True``: quotes rest through the NEXT hour's flow, after the
    crowd has repriced. This is the worst case for a slow bot. A real 60 s
    loop sits between the two.
    """
    world = World(wp, seed)
    strat = make_strategy()
    acct = Account(initial_cash)
    curve: List[float] = []
    resting: Dict[str, Dict[str, Tuple[float, int]]] = {}   # quotes carried into the next hour

    def run_flow(quotes):
        """This hour's takers, with our quotes in the book."""
        for m in world.markets.values():
            for e in m.exchange_ids:
                q = quotes.get(e, {})
                fb, fa = world.taker_flow(m, e, q.get("bid"), q.get("ask"))
                if fb:
                    acct.buy(world.t, e, "yes", q["bid"][0], fb, "quote")
                if fa:
                    acct.buy(world.t, e, "no", round(1 - q["ask"][0], 3), fa, "quote")

    while True:
        if world.t >= wp.hours:
            # Wind-down: stop new listings and run until everything settles,
            # so the final number is realised PnL rather than a mark.
            world.wp = replace(world.wp, market_arrival_per_day=0.0)
            if not world.markets:
                break
        world.advance()

        # ---- settle markets that just expired ----
        for m in world.settle_due():
            for e in m.exchange_ids:
                acct.settle(e, m.pays_yes(e))
                resting.pop(e, None)

        ex_market = {e: m for m in world.markets.values() for e in m.exchange_ids}
        if stale_quotes:
            # Last hour's quotes meet the repriced book. A quote that now
            # crosses the book trades straight away at the book's prices
            # (we are the one picked off), and the rest meets the takers.
            for e, q in resting.items():
                bids, asks = ex_market[e].books[e]
                if "bid" in q:
                    px, qty = q["bid"]
                    n = execute_against_book(acct, world.t, OrderIntent(e, "yes", px, qty, "quote"), bids, asks)
                    q["bid"] = (px, qty - n)
                if "ask" in q:
                    px, qty = q["ask"]
                    n = execute_against_book(acct, world.t, OrderIntent(e, "no", round(1 - px, 3), qty, "quote"), bids, asks)
                    q["ask"] = (px, qty - n)
            run_flow(resting)

        # ---- build the strategy's view ----
        groups: List[GroupView] = []
        marks: Dict[str, float] = {}
        for m in world.markets.values():
            views = []
            for e in m.exchange_ids:
                bids, asks = m.books[e]
                views.append(ExchangeView(e, [tuple(x) for x in bids], [tuple(x) for x in asks], m.last_trade[e]))
                strat.observe_price(e, m.last_trade[e])
                if bids and asks:
                    marks[e] = (bids[0][0] + asks[0][0]) / 2
            groups.append(GroupView(m.market_id, m.market_id, views,
                                    exclusive=not m.is_binary, exhaustive=not m.is_binary,
                                    hours_to_settle=m.end - world.t))

        intents = strat.decide(groups, acct.portfolio(marks))

        # ---- execute: marketable orders hit the book now; quotes rest ----
        quotes: Dict[str, Dict[str, Tuple[float, int]]] = {}
        for it in intents:
            bids, asks = ex_market[it.exchange_id].books[it.exchange_id]
            if it.kind == "quote":
                q = quotes.setdefault(it.exchange_id, {})
                if it.side == "yes":
                    q["bid"] = (it.price, it.qty)
                else:
                    q["ask"] = (round(1 - it.price, 3), it.qty)
            else:
                execute_against_book(acct, world.t, it, bids, asks)

        if stale_quotes:
            resting = quotes
        else:
            run_flow(quotes)

        curve.append(acct.equity(marks))

    # Market-level win rate: did each traded market make money?
    pnl_per_market: Dict[str, float] = {}
    settled = {m.market_id: m for m in world.settled}
    for f in acct.fills:
        mid = f.exchange_id.rsplit("-", 1)[0]
        m = settled.get(mid)
        if m is None:
            continue
        payout = 1.0 if (f.side == "yes") == m.pays_yes(f.exchange_id) else 0.0
        pnl_per_market[mid] = pnl_per_market.get(mid, 0.0) + f.qty * (payout - f.price)
    won_markets = sum(1 for v in pnl_per_market.values() if v > 0)

    return RunResult(acct.cash, initial_cash, curve, dict(acct.pnl_by_kind), len(acct.fills),
                     len(pnl_per_market), won_markets)


def summarise(name: str, results: List[RunResult]) -> Dict[str, float]:
    rets = [r.ret for r in results]
    row = {
        "name": name,
        "mean_ret": statistics.mean(rets),
        "sd_ret": statistics.pstdev(rets) if len(rets) > 1 else 0.0,
        "median_ret": statistics.median(rets),
        "worst_ret": min(rets),
        "pct_profitable": sum(r > 0 for r in rets) / len(rets),
        "max_dd": statistics.mean(r.max_drawdown for r in results),
        "sharpe": statistics.mean(r.daily_sharpe for r in results),
        "fills": statistics.mean(r.n_fills for r in results),
        "mkt_win": statistics.mean(r.markets_won / r.markets_traded if r.markets_traded else 0 for r in results),
    }
    for k in ("arb", "take", "quote"):
        row[k] = statistics.mean(r.pnl_by_kind.get(k, 0.0) for r in results)
    return row


def print_table(rows: List[Dict[str, float]]) -> None:
    hdr = (f"{'configuration':<34}{'mean':>8}{'sd':>8}{'median':>8}{'worst':>8}{'%>0':>6}"
           f"{'maxDD':>7}{'Sharpe':>7}{'arb$':>8}{'take$':>8}{'quote$':>8}{'mktWin':>7}")
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['name']:<34}{r['mean_ret']:>7.1%} {r['sd_ret']:>7.1%} {r['median_ret']:>7.1%} "
              f"{r['worst_ret']:>7.1%} {r['pct_profitable']:>5.0%} {r['max_dd']:>6.1%} {r['sharpe']:>6.1f} "
              f"{r['arb']:>7.0f} {r['take']:>7.0f} {r['quote']:>7.0f} {r['mkt_win']:>6.0%}")


def _suite_job(args):
    """One (configuration, seed) run. Top-level so multiprocessing can pickle it."""
    name, kind, params, wp, seed, initial = args
    make = (lambda: RandomTrader(seed)) if kind == "random" else (lambda: Strategy(params))
    return name, run_synthetic(make, wp, seed, initial, stale_quotes=(kind == "stale"))


def synthetic_suite(seeds: int, hours: int, initial: float, workers: int = 1,
                    first_seed: int = 0) -> List[Dict[str, float]]:
    """Full strategy, one-component ablations, parameter variants, stress
    worlds and baselines, all on the same seeds so they compare like for like."""
    wp = WorldParams(hours=hours)
    base = StrategyParams()
    off = dict(arb_enabled=False, take_enabled=False, mm_enabled=False)
    configs = [
        ("FULL STRATEGY", "strategy", base, wp),
        ("  arbitrage only", "strategy", replace(base, **{**off, "arb_enabled": True}), wp),
        ("  taking only", "strategy", replace(base, **{**off, "take_enabled": True}), wp),
        ("  market making only", "strategy", replace(base, **{**off, "mm_enabled": True}), wp),
        ("  no longshot correction (g=1)", "strategy", replace(base, longshot_gamma=1.0), wp),
        ("  full Kelly + loose risk caps", "strategy",
         replace(base, kelly_fraction=1.0, max_exchange_risk_frac=0.25, max_market_risk_frac=0.35), wp),
        ("STRESS: no longshot bias in world", "strategy", base, replace(wp, gamma_true=1.0)),
        ("STRESS: 2x informed takers", "strategy", base, replace(wp, informed_share=0.5)),
        ("STRESS: half the order flow", "strategy", base,
         replace(wp, takers_per_hour_min=0.15, takers_per_hour_max=1.5)),
        ("STRESS: tight competitor spreads", "strategy", base,
         replace(wp, min_half_spread_ticks=1, max_half_spread_ticks=2)),
        ("STRESS: quotes 1 hour stale", "stale", base, wp),
        ("STRESS: stale + no MM (arb+take)", "stale", replace(base, mm_enabled=False), wp),
        ("STRESS: all of the above", "stale", base,
         replace(wp, gamma_true=1.0, informed_share=0.5, takers_per_hour_min=0.15,
                 takers_per_hour_max=1.5, min_half_spread_ticks=1, max_half_spread_ticks=2)),
        ("BASELINE: random taker", "random", base, wp),
        ("BASELINE: do nothing", "strategy", replace(base, **off), wp),
    ]
    tasks = [(name, kind, params, w, seed, initial)
             for name, kind, params, w in configs
             for seed in range(first_seed, first_seed + seeds)]
    if workers > 1:
        from multiprocessing import Pool
        with Pool(workers) as pool:
            out = pool.map(_suite_job, tasks)
    else:
        out = [_suite_job(t) for t in tasks]
    return [summarise(name, [r for n, r in out if n == name]) for name, *_ in configs]


# ----------------------------------------------------------------------------
# Real-data replay
# ----------------------------------------------------------------------------

def replay_history(path: str, params: StrategyParams, initial: float = 10_000.0,
                   half_spread: float = 0.01, volume_share: float = 0.2) -> RunResult:
    """Replay settled markets saved by ``history.py download``.

    For each candle, we build a pseudo-book around
    the close: bid = close - half_spread, ask = close + half_spread, with depth
    equal to ``volume_share`` of that candle's volume (minimum 10 shares).
    Taking fills at those prices. A resting quote fills only if the NEXT
    candle's range trades strictly through it. That is pessimistic, but
    candles cannot tell us our place in the queue.

    The candle size sets how long a quote rests before it can be revised. With
    hourly candles every quote is an hour stale, and the crowd's drift
    adversely selects it, a much harsher world than a bot that re-quotes every
    minute. Use 1m or 5m candles to judge market making.
    """
    with open(path) as f:
        data = json.load(f)
    strat = Strategy(params)
    acct = Account(initial)

    # Index candles by timestamp.
    series: Dict[str, Dict[str, dict]] = {}
    meta: Dict[str, dict] = {}
    times = set()
    for m in data["markets"]:
        if m.get("settledWith") in (None, "REFUND"):
            continue
        meta[m["id"]] = m
        for ex in m["exchanges"]:
            c = {cd["time"]: cd for cd in ex.get("candles", []) if cd.get("close") is not None}
            series[ex["id"]] = c
            times.update(c)
    order = sorted(times)
    settle_at = {mid: m.get("settlementDate") or max((t for ex in m["exchanges"] for t in series.get(ex["id"], {})), default="")
                 for mid, m in meta.items()}
    settled = set()
    last_close: Dict[str, float] = {}
    last_vol: Dict[str, float] = {}
    curve: List[float] = []
    pending_quotes: Dict[str, Dict[str, Tuple[float, int]]] = {}

    def yes_wins(m: dict, ex: dict) -> bool:
        sw = str(m["settledWith"]).strip().lower()
        if not m.get("isMultiOutcome") and len(m["exchanges"]) == 1:
            return sw in ("yes", "true", "1")
        return sw in (str(ex.get("option", "")).strip().lower(), str(ex["id"]).lower())

    for i, t in enumerate(order):
        # 1. Fill quotes left resting from the previous step, using this candle's range.
        for ex_id, q in pending_quotes.items():
            cd = series.get(ex_id, {}).get(t)
            if not cd:
                continue
            cap = max(10, int(volume_share * (cd.get("volume") or 0)))
            if "bid" in q and cd["low"] < q["bid"][0] - 1e-9:
                acct.buy(i, ex_id, "yes", q["bid"][0], min(q["bid"][1], cap), "quote")
            if "ask" in q and cd["high"] > q["ask"][0] + 1e-9:
                acct.buy(i, ex_id, "no", round(1 - q["ask"][0], 3), min(q["ask"][1], cap), "quote")
        pending_quotes = {}

        # 2. Settle markets whose settlement time has passed.
        for mid, m in meta.items():
            if mid not in settled and settle_at[mid] and t >= settle_at[mid]:
                for ex in m["exchanges"]:
                    acct.settle(ex["id"], yes_wins(m, ex))
                settled.add(mid)

        # 3. Build pseudo-books for every live exchange. Candles exist only when
        #    something traded, so between trades the last close carries forward.
        for ex_id, cs in series.items():
            cd = cs.get(t)
            if cd:
                last_close[ex_id] = cd["close"]
                last_vol[ex_id] = cd.get("volume") or 0
        groups, marks, books = [], {}, {}
        for mid, m in meta.items():
            if mid in settled:
                continue
            views = []
            for ex in m["exchanges"]:
                c = last_close.get(ex["id"])
                if c is None:
                    continue
                depth = max(10, int(volume_share * last_vol[ex["id"]]))
                bid = max(TICK, math.floor((c - half_spread) / TICK) * TICK)
                ask = min(1 - TICK, math.ceil((c + half_spread) / TICK) * TICK)
                bids, asks = [[round(bid, 3), depth]], [[round(ask, 3), depth]]
                books[ex["id"]] = (bids, asks)
                if ex["id"] in series and t in series[ex["id"]]:
                    strat.observe_price(ex["id"], c)
                marks[ex["id"]] = c
                views.append(ExchangeView(ex["id"], [tuple(x) for x in bids], [tuple(x) for x in asks], c))
            if not views:
                continue
            complete = len(views) == len(m["exchanges"])
            hrs = _hours_between(t, settle_at[mid])
            groups.append(GroupView(mid, mid, views,
                                    exclusive=bool(m.get("exclusive")) and complete,
                                    exhaustive=bool(m.get("exhaustive")) and complete,
                                    hours_to_settle=hrs))

        # 4. Decide and execute.
        for it in strat.decide(groups, acct.portfolio(marks)):
            bids, asks = books[it.exchange_id]
            if it.kind == "quote":
                q = pending_quotes.setdefault(it.exchange_id, {})
                if it.side == "yes":
                    q["bid"] = (it.price, it.qty)
                else:
                    q["ask"] = (round(1 - it.price, 3), it.qty)
            else:
                execute_against_book(acct, i, it, bids, asks)
        curve.append(acct.equity(marks))

    # Anything still open at the end of the file settles on its recorded result.
    for mid, m in meta.items():
        if mid not in settled:
            for ex in m["exchanges"]:
                acct.settle(ex["id"], yes_wins(m, ex))
    curve.append(acct.cash)
    traded = {f.exchange_id for f in acct.fills}
    return RunResult(acct.cash, initial, curve, dict(acct.pnl_by_kind), len(acct.fills), len(traded), 0)


def _hours_between(a: str, b: str) -> float:
    from datetime import datetime
    try:
        fa = datetime.fromisoformat(a.replace("Z", "+00:00"))
        fb = datetime.fromisoformat(b.replace("Z", "+00:00"))
        return (fb - fa).total_seconds() / 3600
    except (ValueError, AttributeError):
        return float("inf")


# ----------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seeds", type=int, default=20, help="Monte Carlo runs per configuration")
    ap.add_argument("--days", type=int, default=60, help="trading days per synthetic run (before wind-down)")
    ap.add_argument("--cash", type=float, default=10_000.0, help="starting balance")
    ap.add_argument("--real", metavar="HISTORY_JSON", help="replay real settled markets instead")
    ap.add_argument("--gamma", type=float, default=None, help="override longshot_gamma (e.g. from history.py calibrate)")
    ap.add_argument("--half-spread", type=float, default=0.01, help="assumed half-spread in --real mode")
    ap.add_argument("--workers", type=int, default=1, help="parallel processes for the synthetic suite")
    args = ap.parse_args()

    params = StrategyParams()
    if args.gamma is not None:
        params = replace(params, longshot_gamma=args.gamma)

    if args.real:
        rows = []
        for name, pr in [("FULL STRATEGY", params),
                         ("  taking only", replace(params, arb_enabled=False, mm_enabled=False)),
                         ("  market making only", replace(params, arb_enabled=False, take_enabled=False)),
                         ("  no longshot correction", replace(params, longshot_gamma=1.0))]:
            r = replay_history(args.real, pr, args.cash, args.half_spread)
            rows.append(summarise(name, [r]))
        print_table(rows)
        return

    print(f"Synthetic backtest: {args.seeds} seeds x {args.days} days, starting cash {args.cash:,.0f}")
    rows = synthetic_suite(args.seeds, args.days * 24, args.cash, args.workers)
    print()
    print_table(rows)
    print("\nmean/sd/median/worst = total return over the run; %>0 = share of seeds that made money;")
    print("maxDD = mean peak-to-trough drawdown; arb$/take$/quote$ = mean realised PnL per component;")
    print("mktWin = share of traded markets that ended in profit.")


if __name__ == "__main__":
    main()
