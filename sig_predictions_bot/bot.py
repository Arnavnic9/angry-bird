"""Live trading loop for the SIG Predictions Cup.

    export SUPER_MARKET_API_KEY=...            # scopes: read + trade
    python -m sig_predictions_bot.bot --tournament <slug>           # dry run: prints orders only
    python -m sig_predictions_bot.bot --tournament <slug> --live    # places real orders

Every cycle (default 60 s):

  1. Portfolio. Read cash (tournament ``myBalance``) and positions. Equity is
     cash plus the market value of positions.
  2. Screen. One bulk price call per 100 exchanges. Rank markets by how much
     opportunity the touch shows: a wide spread to quote into, a basket whose
     prices don't sum to 1, or a position we need to manage.
  3. Books. Fetch full order books for the top ``--max-books`` markets only.
     This is how the bot stays inside 100 reads/min.
  4. Decide. Pass everything to ``Strategy.decide``, the same code the
     backtest runs.
  5. Execute. Cancel all our resting orders (1 write), send arbitrage baskets
     as atomic multi-leg orders, and send everything else in batches of up to
     50 (1 write each). About 3-6 writes per cycle, well under 30/min.

Safety: dry-run is the default. A drawdown circuit breaker cancels
everything and stops if equity falls ``--max-drawdown`` below its peak.
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from .client import ApiError, SuperMarketClient
from .config import StrategyParams
from .strategy import ExchangeView, GroupView, OrderIntent, Portfolio, Position, Strategy


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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tournament", required=True, help="tournament slug (see GET /tournaments)")
    ap.add_argument("--live", action="store_true", help="actually place orders (default: dry run)")
    ap.add_argument("--cycle", type=float, default=60.0, help="seconds per cycle")
    ap.add_argument("--max-books", type=int, default=40, help="full order books fetched per cycle")
    ap.add_argument("--max-cycles", type=int, default=None)
    ap.add_argument("--max-drawdown", type=float, default=0.4, help="stop if equity falls this far below peak")
    ap.add_argument("--gamma", type=float, default=None, help="longshot_gamma from history.py calibrate")
    args = ap.parse_args()

    params = StrategyParams()
    if args.gamma is not None:
        params = replace(params, longshot_gamma=args.gamma)
    bot = Bot(SuperMarketClient(), args.tournament, params, args.live, args.max_books, args.max_drawdown)
    bot.run(args.cycle, args.max_cycles)


if __name__ == "__main__":
    main()
