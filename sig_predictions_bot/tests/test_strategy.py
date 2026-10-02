"""Unit tests. Run with:  python -m unittest discover -s sig_predictions_bot/tests -t ."""

import json
import os
import tempfile
import unittest
from dataclasses import replace

from sig_predictions_bot.backtest import Account, replay_history, run_synthetic
from sig_predictions_bot.config import TICK, StrategyParams
from sig_predictions_bot.history import export_synthetic, fit_gamma
from sig_predictions_bot.simulator import WorldParams
from sig_predictions_bot.strategy import (ExchangeView, GroupView, Portfolio, Position, Strategy,
                                          ceil_tick, floor_tick)


def on_tick(p):
    return abs(p / TICK - round(p / TICK)) < 1e-6


def flat(cash=10_000.0):
    return Portfolio(cash=cash, equity=cash, positions={})


class TickTests(unittest.TestCase):
    def test_rounding(self):
        self.assertEqual(floor_tick(0.4249), 0.42)
        self.assertEqual(ceil_tick(0.4201), 0.425)
        self.assertEqual(floor_tick(0.425), 0.425)


class ArbitrageTests(unittest.TestCase):
    def test_yes_basket_when_asks_sum_below_one(self):
        exs = [ExchangeView(f"e{i}", [(0.25, 100)], [(0.30, 100)]) for i in range(3)]
        g = GroupView("g", "m", exs, exclusive=True, exhaustive=True)
        p = replace(StrategyParams(), take_enabled=False, mm_enabled=False)
        intents = Strategy(p).decide([g], flat())
        self.assertEqual(len(intents), 3)
        self.assertTrue(all(i.kind == "arb" and i.side == "yes" and i.qty == 100 for i in intents))
        self.assertEqual(len({i.basket_id for i in intents}), 1)    # one atomic multi-leg order
        cost = sum(i.price * i.qty for i in intents)
        self.assertLess(cost, 100)                                  # 100 baskets pay exactly 100

    def test_no_basket_when_bids_sum_above_one(self):
        exs = [ExchangeView(f"e{i}", [(0.40, 50)], [(0.45, 50)]) for i in range(3)]
        g = GroupView("g", "m", exs, exclusive=True, exhaustive=True)
        p = replace(StrategyParams(), take_enabled=False, mm_enabled=False)
        intents = Strategy(p).decide([g], flat())
        self.assertTrue(all(i.side == "no" and i.qty == 50 for i in intents))
        # Each basket costs 3 * 0.60 = 1.80 and pays n - 1 = 2.
        self.assertAlmostEqual(sum(i.price for i in intents), 1.80)

    def test_no_yes_arb_unless_exhaustive_or_exclusive(self):
        exs = [ExchangeView(f"e{i}", [(0.25, 100)], [(0.30, 100)]) for i in range(3)]
        p = replace(StrategyParams(), take_enabled=False, mm_enabled=False)
        # Independent outcomes: no basket trade.
        self.assertEqual(Strategy(p).decide([GroupView("g", "m", exs)], flat()), [])
        # Exclusive but not exhaustive: buying every YES is NOT riskless.
        g = GroupView("g", "m", exs, exclusive=True, exhaustive=False)
        self.assertFalse([i for i in Strategy(p).decide([g], flat()) if i.side == "yes"])


class FairValueTests(unittest.TestCase):
    def test_gamma_one_is_identity_on_binary_mid(self):
        s = Strategy(replace(StrategyParams(), longshot_gamma=1.0))
        g = GroupView("g", "m", [ExchangeView("e", [(0.30, 10)], [(0.40, 10)])])
        self.assertAlmostEqual(s.fair_values(g)["e"], 0.35, places=6)

    def test_longshot_correction_direction(self):
        s = Strategy(replace(StrategyParams(), longshot_gamma=1.2))
        lo = s.fair_values(GroupView("a", "m", [ExchangeView("a", [(0.09, 10)], [(0.11, 10)])]))["a"]
        hi = s.fair_values(GroupView("b", "m", [ExchangeView("b", [(0.89, 10)], [(0.91, 10)])]))["b"]
        self.assertLess(lo, 0.10)      # longshot worth less than its price
        self.assertGreater(hi, 0.90)   # favourite worth more

    def test_exhaustive_basket_sums_to_one(self):
        exs = [ExchangeView(f"e{i}", [(b, 10)], [(b + 0.02, 10)]) for i, b in enumerate([0.5, 0.3, 0.25])]
        fv = Strategy().fair_values(GroupView("g", "m", exs, exclusive=True, exhaustive=True))
        self.assertAlmostEqual(sum(fv.values()), 1.0, places=9)

    def test_one_sided_book_does_not_use_the_ask_as_probability(self):
        s = Strategy(replace(StrategyParams(), longshot_gamma=1.0))
        fv = s.fair_values(GroupView("g", "m", [ExchangeView("e", [], [(0.02, 10)])]))
        self.assertAlmostEqual(fv["e"], 0.01, places=6)


class TakingAndQuotingTests(unittest.TestCase):
    def test_take_cheap_yes_within_risk_cap(self):
        p = replace(StrategyParams(), arb_enabled=False, mm_enabled=False, longshot_gamma=1.0)
        # Two-outcome exhaustive market whose prices sum to only 0.72. After
        # normalising, outcome A is worth ~0.85 but offered at 0.62.
        a = ExchangeView("a", [(0.60, 5000)], [(0.62, 5000)])
        b = ExchangeView("b", [(0.10, 5000)], [(0.12, 5000)])
        intents = Strategy(p).decide([GroupView("g", "m", [a, b], exclusive=True, exhaustive=True)], flat())
        buys = [i for i in intents if i.exchange_id == "a" and i.side == "yes"]
        self.assertTrue(buys)
        risk = sum(i.qty * i.price for i in buys)
        self.assertLessEqual(risk, p.max_exchange_risk_frac * 10_000 + 1e-6)

    def test_quotes_on_tick_never_cross_and_bid_below_ask(self):
        s = Strategy(replace(StrategyParams(), arb_enabled=False, take_enabled=False))
        ex = ExchangeView("e", [(0.40, 100)], [(0.50, 100)])
        intents = s.decide([GroupView("g", "m", [ex], hours_to_settle=100)], flat())
        bid = [i.price for i in intents if i.side == "yes"][0]
        ask_yes = [1 - i.price for i in intents if i.side == "no"][0]
        self.assertTrue(on_tick(bid) and on_tick(ask_yes))
        self.assertLess(bid, 0.50)          # rests, doesn't lift the ask
        self.assertGreater(ask_yes, 0.40)   # rests, doesn't hit the bid
        self.assertLess(bid, ask_yes)

    def test_no_quotes_near_settlement(self):
        s = Strategy(replace(StrategyParams(), arb_enabled=False, take_enabled=False))
        ex = ExchangeView("e", [(0.40, 100)], [(0.50, 100)])
        self.assertEqual(s.decide([GroupView("g", "m", [ex], hours_to_settle=1)], flat()), [])

    def test_inventory_skews_quotes_down_when_long(self):
        p = replace(StrategyParams(), arb_enabled=False, take_enabled=False)
        ex = ExchangeView("e", [(0.47, 100)], [(0.53, 100)])
        g = GroupView("g", "m", [ex], hours_to_settle=100)
        quotes = lambda pf: {i.side: i.price for i in Strategy(p).decide([g], pf)}
        flat_q = quotes(flat())
        long_q = quotes(Portfolio(10_000, 10_000, {"e": Position(yes_qty=1000, yes_cost=500)}))
        self.assertLess(long_q["yes"], flat_q["yes"])        # bid lower: buy less eagerly
        self.assertGreater(long_q["no"], flat_q["no"])       # NO price higher = YES ask lower: sell more eagerly


class AccountingTests(unittest.TestCase):
    def test_netting_returns_cash(self):
        a = Account(100)
        a.buy(0, "e", "yes", 0.4, 10, "take")
        a.buy(0, "e", "no", 0.5, 10, "take")   # 10 YES + 10 NO = 10 cash
        self.assertAlmostEqual(a.cash, 100 - 4 - 5 + 10)
        self.assertEqual(a.positions["e"].yes_qty + a.positions["e"].no_qty, 0)

    def test_attribution_sums_to_pnl(self):
        r = run_synthetic(lambda: Strategy(), WorldParams(hours=24 * 10, initial_markets=8), seed=3,
                          initial_cash=5_000)
        self.assertAlmostEqual(sum(r.pnl_by_kind.values()), r.final_equity - r.initial, places=6)
        r = run_synthetic(lambda: Strategy(), WorldParams(hours=24 * 10, initial_markets=8), seed=3,
                          initial_cash=5_000, stale_quotes=True)
        self.assertAlmostEqual(sum(r.pnl_by_kind.values()), r.final_equity - r.initial, places=6)


class PipelineTests(unittest.TestCase):
    def test_export_calibrate_replay(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "h.json")
            export_synthetic(path, seed=5, days=8)
            with open(path) as f:
                data = json.load(f)
            self.assertTrue(data["markets"])
            r = replay_history(path, StrategyParams())
            self.assertAlmostEqual(sum(r.pnl_by_kind.values()), r.final_equity - r.initial, places=6)

    def test_fit_gamma_recovers_bias(self):
        import random
        rng = random.Random(0)
        obs = []
        for _ in range(20000):
            true_p = rng.random()
            g = 1.3   # crowd quotes p^(1/g): longshot bias
            price = true_p ** (1 / g) / (true_p ** (1 / g) + (1 - true_p) ** (1 / g))
            obs.append((price, rng.random() < true_p))
        g_hat, _ = fit_gamma(obs)
        self.assertAlmostEqual(g_hat, 1.3, delta=0.08)


if __name__ == "__main__":
    unittest.main()
