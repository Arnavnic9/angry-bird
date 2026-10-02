# SIG Predictions Cup trading bot

A profit-maximising trading bot for the Super Market prediction-market API
(`https://www.thesuper.market/api/v1`), with a backtester. Pure Python 3.9+
standard library: no `pip install` needed.

```
sig_predictions_bot/
  config.py     every tunable parameter, with the reasoning behind its default
  strategy.py   the decision logic (pure, no I/O); shared by the live bot and the backtester
  client.py     API client: auth, rate limiting (100 reads / 30 writes per min), retries, idempotency
  bot.py        live trading loop (dry-run by default)
  simulator.py  synthetic prediction-market world for backtesting
  backtest.py   synthetic Monte Carlo backtest + replay of real settled markets
  history.py    download real settled-market history; fit the longshot-bias parameter
  tests/        unit tests + an end-to-end test of the live loop against a fake API
```

## What the algorithm does

Every contract is binary: YES pays 1 if the outcome happens, NO pays 1 if it
doesn't. Prices are YES-normalised in [0, 1] and there are **no trading
fees**, so any edge captured is kept. Each cycle the strategy runs three
profit engines, in priority order:

1. **Arbitrage (risk-free).** In a mutually exclusive market at most one
   outcome can win. If the best YES asks sum below 1 (and exactly one outcome
   must win), buying one of each costs under 1 and pays exactly 1. If the best
   YES bids sum above 1, buying one NO of each costs `sum(1 - bid)` and pays
   at least `n - 1`. Legs are sent together as an atomic multi-leg order.
   Exclusivity comes from the engine's `mutually_exclusive` relationships, not
   from guessing.
2. **Taking (positive expected value).** A fair probability is estimated for
   every contract: book mid, overround removed across exhaustive baskets, then
   a **favourite-longshot correction** `q ∝ p^γ` (crowds overpay for
   longshots). If the book is mispriced against that fair value by more than
   `min_take_edge`, the bot crosses the spread, sized with **quarter-Kelly**.
3. **Market making (spread capture).** Where the book is wide, it rests a bid
   below and an ask above fair value, improving the touch by one tick. Quotes:
   - are skewed against inventory (when long, quote lower so it sells more),
   - widen with recent volatility (a stale quote is a free option for others),
   - stop 6 hours before settlement, when late news makes resting orders a
     liability.

**Risk limits:** at most 8% of equity can be lost on any one contract and 15%
on any one market. 10% of equity always stays in cash. The live bot also has
a drawdown circuit breaker.

## Quick start

```bash
# backtest (synthetic, 20 seeds, about 5 min on 4 cores)
python -m sig_predictions_bot.backtest --seeds 20 --workers 4

# tests
python -m unittest discover -s sig_predictions_bot/tests -t .

# live: create an API key with read + trade scopes (My Profile -> API Keys)
export SUPER_MARKET_API_KEY=...
python -m sig_predictions_bot.bot --tournament <slug>          # dry run, prints orders
python -m sig_predictions_bot.bot --tournament <slug> --live   # trades
```

**Before going live, calibrate on real data.** The default γ comes from a
simulation, not from Super Market history:

```bash
python -m sig_predictions_bot.history download --tournament <slug> --out history.json   # 5-minute candles
python -m sig_predictions_bot.history calibrate history.json                            # prints fitted gamma
python -m sig_predictions_bot.backtest --real history.json --gamma <fitted>
python -m sig_predictions_bot.bot --tournament <slug> --gamma <fitted> --live
```

## Backtest results

RESULTS_PLACEHOLDER
