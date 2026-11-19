#!/usr/bin/env python3
"""GTA VI trophy rarity forecast — make.fun gta6-trophy-rarity-forecast-01.

Reads every *.in case in MAKEFUN_INPUT_DIR and writes <id>.out forecasts of
five quantiles per trophy, on the percent scale.

Model (all internal, deterministic, stdlib only):
  * Work in logit space (rates clamped to 0.01..99.99), as the scorer does.
  * Candidate medians per trophy:
      - persistence: last valid rate
      - logit-linear trend vs ln(1+hours since 2026-11-19T00:00Z),
        ridge-shrunk slope, windows 24h / 48h / full history
      - decomposition: earners = rate*players/100; players fitted with a
        saturating ln-time line, earners with a ln-time line on log(earners);
        target rate = E(target)/P(target)
      - damped recent drift: logit slope over the last 12h, halved, capped
  * An internal rolling-origin backtest over the second half of each case's
    history (5 origins x 2 evaluation targets) scores every candidate with
    pinball loss on the logit scale. The final median uses the globally
    best candidate alone when it wins the backtest by >=10%, otherwise a
    0.65/0.35 blend of the top two; a trophy uses its own best candidate
    outright when that candidate beats the global choice on the trophy's
    own backtest points by a clear margin. The decomposition candidate's
    earners slope is heavily damped: raw log-earners extrapolation over a
    long ln-time span badly overshoots in backtests.
  * Spread sigma is calibrated from the standardised residuals of the
    blended predictor in the backtest (robust MAD estimator), scaled by
    sqrt(horizon in days), with a hard floor at the house baseline
    sigma = 0.5 per sqrt-day (protects against discrete page-refresh jumps
    when the visible history looks flat).
  * Every per-trophy computation is wrapped: on any anomaly the exact house
    persistence baseline is used. The program never crashes on a case.
"""

import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

DEFAULT_LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
DEFAULT_Z = [-1.645, -0.674, 0.0, 0.674, 1.645]
HOUSE_SIGMA = 0.5  # logit units per sqrt-day
ANCHOR_ISO = "2026-11-19T00:00:00Z"


# ---------------------------------------------------------------- utilities

def clamp_rate(x):
    return min(99.99, max(0.01, x))


def logit(pct):
    p = clamp_rate(pct)
    return math.log(p / (100.0 - p))


def sigmoid(lg):
    if lg >= 0:
        return 100.0 / (1.0 + math.exp(-lg))
    e = math.exp(lg)
    return 100.0 * e / (1.0 + e)


def round4(x):
    """Round to 4 decimals, half up (matching the starter kit)."""
    scaled = x * 10000.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 10000.0
    return int(value) if float(value).is_integer() else value


def parse_time(s):
    """ISO timestamp -> epoch seconds (float). Raises on garbage."""
    txt = str(s).strip()
    if txt.endswith("Z"):
        txt = txt[:-1] + "+00:00"
    dt = datetime.fromisoformat(txt)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


ANCHOR_TS = parse_time(ANCHOR_ISO)


def hours_since_anchor(ts):
    return (ts - ANCHOR_TS) / 3600.0


def log_time(ts):
    """ln(1 + hours since anchor), floored at 0 for pre-anchor reads."""
    h = hours_since_anchor(ts)
    if h < 0.0:
        h = 0.0
    return math.log1p(h)


def norm_ppf(p):
    """Inverse standard normal CDF (Acklam's rational approximation)."""
    if p <= 0.0:
        return -4.0
    if p >= 1.0:
        return 4.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02,
         -2.759285984469534e+02, 1.383577518672690e+02,
         -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02,
         -1.556989798598866e+02, 6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
         4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01,
         2.445134137142996e+00, 3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
               ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / \
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0)
    q = p - 0.5
    r = q * q
    return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
           (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1.0)


def z_for_levels(levels):
    if list(levels) == DEFAULT_LEVELS:
        return list(DEFAULT_Z)
    return [norm_ppf(float(p)) for p in levels]


def pinball_logit(actual_logit, median_logit, spread, taus, zs):
    """Mean pinball loss on the logit scale for one observation."""
    total = 0.0
    for tau, z in zip(taus, zs):
        q = median_logit + z * spread
        diff = actual_logit - q
        total += max(tau * diff, (tau - 1.0) * diff)
    return total / max(1, len(taus))


def median_of(vals):
    s = sorted(vals)
    n = len(s)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return s[mid]
    return 0.5 * (s[mid - 1] + s[mid])


# ------------------------------------------------------------- regression

def lin_fit(xs, ys, ridge=1e-6):
    """Least squares y = a + b*x with a tiny ridge on the slope.

    Returns (a, b) or None when the fit is not identifiable.
    """
    n = len(xs)
    if n == 0:
        return None
    if n == 1:
        return (ys[0], 0.0)
    mx = sum(xs) / n
    my = sum(ys) / n
    sxx = 0.0
    sxy = 0.0
    for x, y in zip(xs, ys):
        dx = x - mx
        sxx += dx * dx
        sxy += dx * (y - my)
    if sxx <= 1e-12:
        return (my, 0.0)
    b = sxy / (sxx + ridge * max(1.0, sxx))
    a = my - b * mx
    return (a, b)


# ------------------------------------------------------------- case prep

class Series:
    """Compressed per-case data.

    times: epoch seconds per read (sorted, duplicates merged by keeping the
    last read at that timestamp). players: players_tracked per read.
    rates[i]: list of (read_index, rate) with non-null rates for trophy i,
    compressed to drop consecutive duplicate rates at unchanged player
    counts (the page is frozen between refreshes).
    """

    def __init__(self, case):
        hist = case.get("history") or []
        raw = []
        for h in hist:
            try:
                ts = parse_time(h.get("at") or h.get("slot"))
            except Exception:
                continue
            raw.append((ts, h))
        raw.sort(key=lambda t: t[0])
        self.times = []
        self.players = []
        merged = []
        for ts, h in raw:
            if merged and abs(ts - merged[-1][0]) < 0.5:
                merged[-1] = (ts, h)
            else:
                merged.append((ts, h))
        for ts, h in merged:
            self.times.append(ts)
            try:
                p = float(h.get("players_tracked"))
            except Exception:
                p = 0.0
            self.players.append(p)
        self.n_reads = len(self.times)
        n_trophies = len(case.get("trophies") or [])
        self.rates = []
        for i in range(n_trophies):
            pts = []
            prev_key = None
            for j, (ts, h) in enumerate(merged):
                rates = h.get("rates") or []
                if i >= len(rates):
                    continue
                r = rates[i]
                if r is None:
                    continue
                try:
                    r = float(r)
                except Exception:
                    continue
                if not math.isfinite(r):
                    continue
                key = (r, self.players[j])
                if pts and key == prev_key:
                    # Frozen page: same rate at same player count adds no
                    # information, but keep the timestamp fresh by moving
                    # the previous point's read index forward.
                    pts[-1] = (j, pts[-1][1])
                    prev_key = key
                    continue
                pts.append((j, r))
                prev_key = key
            self.rates.append(pts)

    def last_rate(self, i):
        pts = self.rates[i] if i < len(self.rates) else []
        return pts[-1][1] if pts else None

    def rate_at_or_before(self, i, read_idx):
        """Last known rate for trophy i at or before read_idx, or None."""
        val = None
        for (j, r) in self.rates[i]:
            if j <= read_idx:
                val = r
            else:
                break
        return val


# ------------------------------------------------------------- candidates
# Each candidate maps (series, trophy_idx, upto_read_idx, target_ts) to a
# predicted median in logit space, or None when it cannot be computed.
# upto_read_idx is exclusive: only reads with index < upto are used.

def _pts_upto(series, i, upto):
    return [(j, r) for (j, r) in series.rates[i] if j < upto]


def cand_persist(series, i, upto, target_ts):
    pts = _pts_upto(series, i, upto)
    if not pts:
        return None
    return logit(pts[-1][1])


def _trend(series, i, upto, target_ts, window_hours):
    pts = _pts_upto(series, i, upto)
    if len(pts) < 2:
        return None
    if window_hours is not None:
        cutoff = series.times[upto - 1] - window_hours * 3600.0
        pts = [(j, r) for (j, r) in pts if series.times[j] >= cutoff]
        if len(pts) < 2:
            return None
    xs = [log_time(series.times[j]) for (j, _) in pts]
    ys = [logit(r) for (_, r) in pts]
    fit = lin_fit(xs, ys, ridge=0.02)
    if fit is None:
        return None
    a, b = fit
    # Shrink the slope a little: ln-time trends flatten as the cohort matures.
    b *= 0.85
    pred = a + b * log_time(target_ts)
    last = ys[-1]
    # Never extrapolate absurdly far from the last observed logit.
    pred = min(last + 2.5, max(last - 2.5, pred))
    return pred


def cand_trend24(series, i, upto, target_ts):
    return _trend(series, i, upto, target_ts, 24.0)


def cand_trend48(series, i, upto, target_ts):
    return _trend(series, i, upto, target_ts, 48.0)


def cand_trend_all(series, i, upto, target_ts):
    return _trend(series, i, upto, target_ts, None)


def cand_decomp(series, i, upto, target_ts):
    pts = _pts_upto(series, i, upto)
    if len(pts) < 2:
        return None
    # Players: saturating ln-time line over reads that carry a player count.
    px, py = [], []
    prev = None
    for j in range(upto):
        p = series.players[j]
        if p and p > 0:
            key = (p, log_time(series.times[j]))
            if key != prev:
                px.append(log_time(series.times[j]))
                py.append(p)
                prev = key
    if len(px) < 2:
        return None
    pfit = lin_fit(px, py, ridge=0.02)
    if pfit is None:
        return None
    pa, pb = pfit
    p_target = pa + pb * log_time(target_ts)
    p_last = py[-1]
    # Player counts do not shrink on a refresh in any material way.
    p_target = max(p_target, p_last * 0.98, 1.0)
    # Earners: ln-time line on log(earners); zero-earner trophies fall back.
    ex, ey = [], []
    for (j, r) in pts:
        p = series.players[j]
        if p and p > 0:
            e = r * p / 100.0
            if e > 0.05:
                ex.append(log_time(series.times[j]))
                ey.append(math.log(e))
    if len(ex) < 2:
        return None
    efit = lin_fit(ex, ey, ridge=0.02)
    if efit is None:
        return None
    ea, eb = efit
    # Damp the earners slope hard: log-earners extrapolation over a long
    # ln-time span explodes, and backtests showed the raw version badly
    # overshooting. The decomposition may only nudge the last rate.
    eb = min(3.0, max(-1.0, eb)) * 0.6
    e_target = math.exp(ea + eb * log_time(target_ts))
    rate = 100.0 * e_target / p_target
    pred = logit(rate)
    last = logit(pts[-1][1])
    return min(last + 1.5, max(last - 1.5, pred))


def cand_drift12(series, i, upto, target_ts):
    pts = _pts_upto(series, i, upto)
    if len(pts) < 2:
        return None
    t_last = series.times[pts[-1][0]]
    cutoff = t_last - 12.0 * 3600.0
    win = [(j, r) for (j, r) in pts if series.times[j] >= cutoff]
    if len(win) < 2:
        win = pts[-2:]
    xs = [(series.times[j] - t_last) / 3600.0 for (j, _) in win]
    ys = [logit(r) for (_, r) in win]
    fit = lin_fit(xs, ys, ridge=0.05)
    if fit is None:
        return None
    _, slope = fit  # logit per hour
    horizon_h = max(0.0, (target_ts - t_last) / 3600.0)
    delta = 0.5 * slope * horizon_h
    delta = min(1.5, max(-1.5, delta))
    return ys[-1] + delta


CANDIDATES = [
    ("persist", cand_persist),
    ("trend24", cand_trend24),
    ("trend48", cand_trend48),
    ("trend_all", cand_trend_all),
    ("decomp", cand_decomp),
    ("drift12", cand_drift12),
]
CAND_FN = dict(CANDIDATES)


# ------------------------------------------------------------- backtest

def backtest(series, n_trophies, taus, zs):
    """Rolling-origin evaluation over the second half of the history.

    Origins sit at fixed fractions of the read timeline; each origin is
    scored against two later reads (midway and final). Returns a dict with
    global mean loss per candidate, per-trophy mean loss per candidate,
    and the records needed to calibrate the spread.
    """
    result = {"global": {}, "per_trophy": {}, "records": []}
    n = series.n_reads
    if n < 8:
        return result
    records = []  # (trophy_idx, cand_name, pred_logit, actual_logit, d_days)
    for frac in (0.45, 0.55, 0.65, 0.75, 0.85):
        upto = int(n * frac)
        if upto < 3 or upto >= n:
            continue
        origin_ts = series.times[upto - 1]
        eval_idxs = sorted({min(n - 1, upto + (n - upto) // 2), n - 1})
        for ev in eval_idxs:
            if ev < upto:
                continue
            ev_ts = series.times[ev]
            d_days = max(1.0 / 24.0, (ev_ts - origin_ts) / 86400.0)
            for i in range(n_trophies):
                if not any(j < upto for (j, _) in series.rates[i]):
                    continue
                actual = series.rate_at_or_before(i, ev)
                if actual is None:
                    continue
                a_logit = logit(actual)
                for name, fn in CANDIDATES:
                    try:
                        pred = fn(series, i, upto, ev_ts)
                    except Exception:
                        pred = None
                    if pred is None or not math.isfinite(pred):
                        continue
                    records.append((i, name, pred, a_logit, d_days))
    # Aggregate losses with the house spread at each record's horizon.
    glob = {}
    per = {}
    for (i, name, pred, a_logit, d_days) in records:
        loss = pinball_logit(a_logit, pred,
                             HOUSE_SIGMA * math.sqrt(d_days), taus, zs)
        glob.setdefault(name, []).append(loss)
        per.setdefault(i, {}).setdefault(name, []).append(loss)
    result["global"] = {k: sum(v) / len(v) for k, v in glob.items()
                        if len(v) >= 10}
    result["per_trophy"] = {
        i: {k: sum(v) / len(v) for k, v in d.items() if len(v) >= 3}
        for i, d in per.items()}
    result["records"] = records
    result["counts"] = {k: len(v) for k, v in glob.items()}
    return result


def choose_global(best_pool):
    """Rank candidates by global backtest loss; return up to 2 names."""
    scored = sorted(best_pool.items(), key=lambda kv: kv[1])
    return [name for name, _ in scored[:2]]


# ------------------------------------------------------------- forecast

def baseline_row(rate, spread, zs):
    lg = logit(rate if rate is not None else 0.0)
    return [round4(clamp_rate(sigmoid(lg + z * spread))) for z in zs]


def forecast_case(case):
    levels = case.get("levels") or DEFAULT_LEVELS
    taus = [float(p) for p in levels]
    zs = z_for_levels(levels)
    trophies = case.get("trophies") or []
    horizon_min = float(case.get("horizon_minutes") or 0.0)
    d_days = max(horizon_min, 1.0) / 1440.0
    house_spread = HOUSE_SIGMA * math.sqrt(d_days)
    try:
        target_ts = parse_time(case.get("target"))
    except Exception:
        target_ts = None

    series = Series(case)
    if target_ts is None:
        rows = []
        for i, t in enumerate(trophies):
            rows.append({"id": t["id"],
                         "q": baseline_row(series.last_rate(i),
                                           house_spread, zs)})
        return {"trophies": rows}

    bt = backtest(series, len(trophies), taus, zs)
    if bt["global"]:
        ranked = choose_global(bt["global"])
        # Winner-take-all when the best candidate wins the backtest by a
        # clear margin (>=10% lower loss than the runner-up): blending in
        # a clearly worse candidate only dilutes the median. Otherwise
        # blend the top two 0.65/0.35.
        if len(ranked) == 2 and bt["global"][ranked[0]] <= 0.90 * bt["global"][ranked[1]]:
            best_names = [ranked[0]]
            weights = [1.0]
        else:
            best_names = ranked
            weights = [0.65, 0.35][:len(best_names)]
    else:
        best_names = ["persist"]
        weights = [1.0]

    # Calibrate sigma from the effective predictor's standardised
    # residuals in the backtest, with a robust (MAD) scale estimate.
    sigma = HOUSE_SIGMA
    if bt["records"]:
        by_key = {}
        for (i, name, pred, a_logit, d_days) in bt["records"]:
            by_key.setdefault((i, d_days, a_logit), {})[name] = pred
        resids = []
        for (i, d_days, a_logit), preds in by_key.items():
            num = 0.0
            den = 0.0
            for name, w in zip(best_names, weights):
                if name in preds:
                    num += w * preds[name]
                    den += w
            if den > 0:
                resids.append((a_logit - num / den) / math.sqrt(d_days))
        if len(resids) >= 10:
            med = median_of(resids)
            mad = median_of([abs(r - med) for r in resids])
            est = 1.4826 * mad
            if math.isfinite(est) and est > 0:
                sigma = min(1.25, max(HOUSE_SIGMA, est))
    spread = max(house_spread, sigma * math.sqrt(d_days))

    upto = series.n_reads
    rows = []
    sorted_levels = all(taus[k] <= taus[k + 1] for k in range(len(taus) - 1))
    for i, t in enumerate(trophies):
        last = series.last_rate(i)
        try:
            # Per-trophy override: use the trophy's own best candidate when
            # it beats both global-blend members on its own backtest points
            # by at least 15%.
            own = bt["per_trophy"].get(i) or {}
            chosen = None
            if own:
                own_best = min(own.items(), key=lambda kv: kv[1])
                blend_members = [own[n] for n in best_names if n in own]
                if blend_members and own_best[0] not in best_names:
                    if own_best[1] <= 0.85 * min(blend_members):
                        chosen = own_best[0]
            if chosen is not None:
                p = CAND_FN[chosen](series, i, upto, target_ts)
                if p is None or not math.isfinite(p):
                    raise ValueError("chosen candidate failed")
                median_logit = p
            else:
                preds = []
                ws = []
                for name, w in zip(best_names, weights):
                    p = CAND_FN[name](series, i, upto, target_ts)
                    if p is not None and math.isfinite(p):
                        preds.append(p)
                        ws.append(w)
                if not preds:
                    raise ValueError("no candidate")
                wsum = sum(ws)
                median_logit = sum(p * w for p, w in zip(preds, ws)) / wsum
            if last is not None:
                ll = logit(last)
                median_logit = min(ll + 3.0, max(ll - 3.0, median_logit))
            qs = [round4(clamp_rate(sigmoid(median_logit + z * spread)))
                  for z in zs]
            if sorted_levels:
                for k in range(1, len(qs)):
                    if qs[k] < qs[k - 1]:
                        qs[k] = qs[k - 1]
            rows.append({"id": t["id"], "q": qs})
        except Exception:
            rows.append({"id": t["id"],
                         "q": baseline_row(last, house_spread, zs)})
    return {"trophies": rows}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    if not INPUT_DIR.is_dir():
        return
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case)
        except Exception:
            # Last-resort: emit a neutral valid file if the case id is known.
            try:
                case = json.loads(path.read_text(encoding="utf-8"))
                rows = [{"id": t["id"], "q": [0.01, 0.25, 1.0, 4.0, 25.0]}
                        for t in case.get("trophies") or []]
                forecast = {"trophies": rows}
            except Exception:
                forecast = {"trophies": []}
        out = json.dumps(forecast, separators=(",", ":")) + "\n"
        (OUTPUT_DIR / (path.stem + ".out")).write_text(out, encoding="utf-8")


if __name__ == "__main__":
    main()
