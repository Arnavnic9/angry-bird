"""Download settled-market history and calibrate the strategy on it.

    # 1. Download settled markets of a tournament, with 5-minute candles, to JSON
    python -m sig_predictions_bot.history download --tournament <slug> --out history.json

    # 2. Fit the favourite-longshot parameter gamma on that data
    python -m sig_predictions_bot.history calibrate history.json

    # 3. Backtest on the real data with the fitted gamma
    python -m sig_predictions_bot.backtest --real history.json --gamma <fitted>

    # (offline) write a synthetic file in the same format, to test the pipeline
    python -m sig_predictions_bot.history export-synthetic --out synthetic_history.json

Calibration
-----------
For every settled exchange we sample the candle close once per day, then ask
which gamma in q = p^g / (p^g + (1-p)^g) best predicts the actual outcomes
(maximum log-likelihood). gamma > 1 means longshots were overpriced and
favourites underpriced, the classic favourite-longshot bias. gamma < 1 means
the crowd was too timid, and the strategy will lean the other way. A
reliability table (price bucket vs actual win frequency) is printed so you can
check the fit by eye.

Fit on older markets and backtest on newer ones, otherwise the backtest has
seen its own answers. Prices within one market share a single outcome, so
the effective sample size is the number of markets, not candles. Expect gamma
to be noisy below a few hundred settled markets.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import List, Tuple

from .simulator import World, WorldParams


# ----------------------------------------------------------------------------
# Download
# ----------------------------------------------------------------------------

def fetch_candles(c, exchange_id: str, tid: str, resolution: str, max_pages: int = 20) -> list:
    """Page forward through an exchange's whole candle history.

    Without ``from`` the API returns only the newest 1000 candles, so we start
    at the epoch and follow the response's ``to`` (the exclusive continuation
    boundary) while ``coverage.complete`` is false.
    """
    out, start = [], "1970-01-01T00:00:00Z"
    for _ in range(max_pages):
        h = c.price_history(exchange_id, tid, resolution=resolution, start=start, limit=1000)
        out.extend(h.get("candles", []))
        if h.get("coverage", {}).get("complete", True) or not h.get("candles"):
            break
        start = h["to"]
    return out


def download(slug: str, out: str, max_markets: int = 500, resolution: str = "5m") -> None:
    from .client import SuperMarketClient

    c = SuperMarketClient()
    tid = c.tournament(slug)["id"]
    markets = c.markets(tid, status="settled")[:max_markets]
    try:
        rels = c.relationships(tid, rel_type="mutually_exclusive")
    except Exception:
        rels = []
    ex_rel = {}
    for r in rels:
        for n in r.get("nodes", []):
            ex_rel[n["exchangeId"]] = bool(r.get("isExhaustive"))

    rows = []
    for i, m in enumerate(markets):
        exs = []
        for e in m.get("exchanges", []):
            exs.append({"id": e["id"], "option": e.get("option"),
                        "candles": fetch_candles(c, e["id"], tid, resolution)})
        ids = [e["id"] for e in exs]
        exclusive = len(ids) > 1 and all(i in ex_rel for i in ids)
        rows.append({
            "id": m["id"], "title": m.get("title"), "settledWith": m.get("settledWith"),
            "settlementDate": m.get("settledOn") or m.get("settlementDate"),
            "isMultiOutcome": m.get("isMultiOutcome", len(ids) > 1),
            "exclusive": exclusive, "exhaustive": exclusive and all(ex_rel[i] for i in ids),
            "exchanges": exs,
        })
        print(f"  [{i + 1}/{len(markets)}] {m.get('title', m['id'])[:60]}")
    with open(out, "w") as f:
        json.dump({"tournament": slug, "resolution": resolution, "markets": rows}, f)
    print(f"saved {len(rows)} settled markets -> {out}")


# ----------------------------------------------------------------------------
# Calibration
# ----------------------------------------------------------------------------

def _yes_wins(m: dict, ex: dict) -> bool:
    sw = str(m.get("settledWith")).strip().lower()
    if len(m["exchanges"]) == 1:
        return sw in ("yes", "true", "1")
    return sw in (str(ex.get("option", "")).strip().lower(), str(ex["id"]).lower())


def observations(data: dict, every_hours: int = 24) -> List[Tuple[float, bool]]:
    step_h = {"1m": 1 / 60, "5m": 5 / 60, "1h": 1, "1d": 24, "1w": 168}.get(data.get("resolution", "1h"), 1)
    every = max(1, int(round(every_hours / step_h)))
    obs = []
    for m in data["markets"]:
        if m.get("settledWith") in (None, "REFUND"):
            continue
        for ex in m["exchanges"]:
            cs = [c for c in ex.get("candles", []) if c.get("close") is not None]
            y = _yes_wins(m, ex)
            for c in cs[::every]:
                obs.append((float(c["close"]), y))
    return obs


def fit_gamma(obs: List[Tuple[float, bool]]) -> Tuple[float, float]:
    """Grid-search maximum likelihood for gamma. Returns (gamma, log-likelihood)."""
    def ll(g):
        s = 0.0
        for p, y in obs:
            p = min(max(p, 1e-3), 1 - 1e-3)
            q = p ** g / (p ** g + (1 - p) ** g)
            s += math.log(q if y else 1 - q)
        return s
    best = max((ll(g / 100), g / 100) for g in range(60, 201, 1))
    return best[1], best[0]


def calibrate(path: str) -> float:
    with open(path) as f:
        obs = observations(json.load(f))
    if len(obs) < 50:
        print(f"only {len(obs)} observations; too few to calibrate reliably")
    g, ll = fit_gamma(obs)
    print(f"observations: {len(obs)}   fitted gamma: {g:.2f}   (1.00 = crowd perfectly calibrated)")
    print(f"{'price bucket':<14}{'n':>7}{'avg price':>11}{'won':>8}")
    edges = [0, .05, .1, .2, .3, .4, .5, .6, .7, .8, .9, .95, 1.0001]
    for lo, hi in zip(edges, edges[1:]):
        sel = [(p, y) for p, y in obs if lo <= p < hi]
        if sel:
            print(f"{lo:>5.2f}-{hi:<8.2f}{len(sel):>7}{sum(p for p, _ in sel) / len(sel):>11.3f}"
                  f"{sum(y for _, y in sel) / len(sel):>8.3f}")
    return g


# ----------------------------------------------------------------------------
# Synthetic export (same JSON shape as ``download``), for offline testing
# ----------------------------------------------------------------------------

def export_synthetic(out: str, seed: int = 123, days: int = 45) -> None:
    from datetime import datetime, timedelta, timezone

    wp = WorldParams(hours=days * 24)
    w = World(wp, seed)
    t0 = datetime(2026, 1, 1, tzinfo=timezone.utc)
    iso = lambda h: (t0 + timedelta(hours=h)).strftime("%Y-%m-%dT%H:%M:%SZ")
    candles = {}
    while w.t < wp.hours or w.markets:
        if w.t >= wp.hours:
            w.wp.market_arrival_per_day = 0.0
        w.advance()
        for m in w.markets.values():
            for e in m.exchange_ids:
                w.taker_flow(m, e, None, None)       # background trading only
                bids, asks = m.books[e]
                mid = (bids[0][0] + asks[0][0]) / 2 if bids and asks else m.crowd_price(m.outcome_index(e))
                trades = m.hour_trades.get(e, [])
                pxs = [p for p, _ in trades] or [mid]
                candles.setdefault(e, []).append({
                    "time": iso(w.t), "open": round(pxs[0], 4), "high": round(max(pxs), 4),
                    "low": round(min(pxs), 4), "close": round(mid, 4),
                    "volume": float(sum(s for _, s in trades)), "tradeCount": len(trades)})
        w.settle_due()
    rows = []
    for m in w.settled:
        exs = [{"id": e, "option": f"Outcome {m.outcome_index(e)}", "candles": candles.get(e, [])}
               for e in m.exchange_ids]
        settled_with = ("YES" if m.winner == 0 else "NO") if m.is_binary else f"Outcome {m.winner}"
        rows.append({"id": m.market_id, "title": f"synthetic {m.market_id}", "settledWith": settled_with,
                     "settlementDate": iso(m.end), "isMultiOutcome": not m.is_binary,
                     "exclusive": not m.is_binary, "exhaustive": not m.is_binary, "exchanges": exs})
    with open(out, "w") as f:
        json.dump({"tournament": "synthetic", "markets": rows}, f)
    print(f"wrote {len(rows)} synthetic settled markets -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("download")
    d.add_argument("--tournament", required=True)
    d.add_argument("--out", default="history.json")
    d.add_argument("--max-markets", type=int, default=500)
    d.add_argument("--resolution", default="5m", choices=["1m", "5m", "1h"],
                   help="candle size; finer candles make the market-making replay far less pessimistic")
    c = sub.add_parser("calibrate")
    c.add_argument("path")
    s = sub.add_parser("export-synthetic")
    s.add_argument("--out", default="synthetic_history.json")
    s.add_argument("--seed", type=int, default=123)
    s.add_argument("--days", type=int, default=45)
    args = ap.parse_args()
    if args.cmd == "download":
        download(args.tournament, args.out, args.max_markets, args.resolution)
    elif args.cmd == "calibrate":
        calibrate(args.path)
    else:
        export_synthetic(args.out, args.seed, args.days)


if __name__ == "__main__":
    main()
