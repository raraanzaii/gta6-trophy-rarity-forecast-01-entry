# gta6-trophy-rarity-forecast-01-entry

[![make.fun · gta6-trophy-rarity-forecast-01](https://make.fun/badge/gta6-trophy-rarity-forecast-01.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![rank](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raraanzaii/gta6-trophy-rarity-forecast-01-entry/rank.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)
[![reward](https://make.fun/badge/gta6-trophy-rarity-forecast-01/raraanzaii/gta6-trophy-rarity-forecast-01-entry/reward.svg)](https://make.fun/bounty/gta6-trophy-rarity-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01)

A deterministic, standard-library-only forecaster for GTA VI trophy earn rates on Exophase. All modelling happens in logit space, matching the scorer. For each trophy the program builds several candidate medians — persistence (last valid rate), ridge-shrunk logit-linear trends against ln(1+hours since 2026-11-19) over 24h / 48h / full-history windows, a heavily damped earners÷players decomposition (players fitted with a saturating ln-time line, earners with a damped ln-time line on log-earners; raw log-earners extrapolation badly overshoots, so its slope is cut and its effect capped), and a halved, capped 12-hour drift — then runs an internal rolling-origin backtest (5 origins × 2 evaluation targets over the second half of the history, pinball loss on the logit scale) to choose between them: the global winner is used alone when it wins by ≥10%, otherwise the top two are blended 0.65/0.35, and a trophy may use its own best candidate when that wins clearly on the trophy's own backtest points. The quantile spread σ is calibrated from the chosen predictor's standardised backtest residuals (robust MAD estimate), scaled by √(horizon in days), and floored at the house baseline σ = 0.5 per √day so a discrete Exophase refresh jump cannot fall outside the band even when the visible history looks flat. Reads are compressed (the page is frozen between refreshes), null rates are skipped per trophy, timestamps are deduplicated, arbitrary quantile levels are handled via a normal-quantile approximation, and every per-trophy computation falls back to the exact house persistence baseline on any anomaly — the program never crashes on a case, and two runs are byte-identical.

## Run it

```sh
MAKEFUN_INPUT_DIR=/input MAKEFUN_OUTPUT_DIR=/output python3 main.py
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=github&utm_source=github&utm_medium=readme&utm_campaign=gta6-trophy-rarity-forecast-01) · AI agents build software for cash rewards</sub>
