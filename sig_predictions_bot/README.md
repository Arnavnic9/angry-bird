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

`python -m sig_predictions_bot.backtest --seeds 20 --workers 4`: 20 random
worlds, 60 trading days each, plus a wind-down until every market settles,
starting from 10,000. All PnL is realised at settlement. The per-component
PnL sums exactly to the change in equity (checked by a unit test).

```
configuration                         mean      sd  median   worst   %>0  maxDD Sharpe    arb$   take$  quote$ mktWin
---------------------------------------------------------------------------------------------------------------------
FULL STRATEGY                      180.3%   39.3%  177.3%   86.7%  100%   5.8%   15.9       3     -79   18108    80%
  arbitrage only                     0.0%    0.0%    0.0%    0.0%   70%   1.2%    0.1       3       0       0    70%
  taking only                       -0.5%    3.0%    0.0%   -7.4%   35%   2.8%   -0.0       0     -54       0    51%
  market making only               181.6%   37.4%  177.1%   95.6%  100%   5.6%   16.2       0       0   18162    80%
  no longshot correction (g=1)     114.9%   30.4%  111.9%   69.9%  100%   5.2%   12.0       3       0   11491    60%
  full Kelly + loose risk caps     155.6%   47.9%  142.4%   77.6%  100%  13.8%    9.4       3    -112   15667    78%
STRESS: no longshot bias in world  136.4%   34.9%  140.7%   80.8%  100%   8.3%   12.1       2    -183   13819    77%
STRESS: 2x informed takers         131.1%   31.1%  131.2%   67.6%  100%   6.7%   13.7       6     121   12981    80%
STRESS: half the order flow        101.3%   18.3%   96.8%   75.9%  100%   6.0%   12.7       4     147    9975    79%
STRESS: tight competitor spreads    27.4%   18.3%   21.9%    0.9%  100%  11.6%    3.5      10     -90    2820    61%
STRESS: quotes 1 hour stale          9.5%   27.9%   12.1%  -48.7%   65%  23.1%    1.1       3     -71    1021    64%
STRESS: stale + no MM (arb+take)    -0.6%    3.0%    0.0%   -7.4%   45%   3.6%   -0.1       3     -61       0    67%
STRESS: all of the above           -72.4%    8.5%  -74.0%  -87.6%    0%  72.8%  -10.9      10     -56   -7197    31%
BASELINE: random taker             -19.8%   13.4%  -21.1%  -52.3%   10%  26.0%   -2.6       0   -1981       0    39%
BASELINE: do nothing                 0.0%    0.0%    0.0%    0.0%    0%   0.0%    0.0       0       0       0     0%
```

**These numbers come from a simulator and are not a forecast of real
returns.** The simulated world (`simulator.py`) has a noisy, lagging,
longshot-biased crowd, background market makers, and a mix of noise and
informed takers. Its parameters are my assumptions, not measurements of
Super Market. Here is what the table does and doesn't support.

**What the results support:**
- **Market making is the profit engine.** It accounts for almost all the
  PnL. With no fees, capturing the spread from noise traders adds up.
- **The longshot correction adds value.** It lifts returns from +115% to
  +180%, mostly by skewing quotes the right way.
- **Arbitrage is rare but risk-free.** Other traders remove most basket
  mispricing, so it earns little in the simulator. It costs nothing to run,
  and on a real platform with thin liquidity it may fire more often.
- **Pure taking is roughly breakeven.** That is why its edge threshold is
  high (0.06). Raise it or switch it off (`take_enabled=False`) if real data
  agrees.
- **Quarter-Kelly beats full Kelly.** Full Kelly earns less and has more than
  twice the drawdown.

**What breaks it:**
- **Stale quotes are the biggest risk.** If quotes rest an hour while
  prices move, market-making profit drops from +180% to +10%, and the worst
  seed loses 49%. The live bot re-quotes every 60 s (`--cycle`), and a
  shorter cycle is safer. Combine stale quotes with tight competition, heavy
  informed flow and no longshot bias, and it loses badly (-72%).
- **Competition shrinks profit.** If other bots already quote 1-2 ticks
  wide, profit falls to about +27%.

**Real-data replay.** `backtest --real` replays settled markets from
candles. Candles can't show queue position, so a quote counts as filled
only when the price trades through it. That is pessimistic, and with
hourly candles every quote is also an hour stale. On synthetic hourly
exports, market making loses 17-26% in this mode. Use 5-minute candles (the
`history.py` default) to judge it. Run the replay and gamma calibration on
real tournament history before trusting any default.

## Known limitations

- Prices are refreshed by REST polling. Realtime WebSocket order-book feeds
  (`POST /realtime/token`) would cut quote staleness further and are the
  natural next upgrade.
- Only baskets that cover exactly one whole market are traded as baskets.
  Cross-market relationships are ignored.
- Arbitrage legs are limit orders. If the book moves between the read and
  the order, a leg can rest partly unfilled. The next cycle's cancel-all
  clears it, and any leftover counts as a normal position under the risk
  limits.
- The gamma fit is noisy below a few hundred settled markets, because
  candles within one market share a single outcome.
