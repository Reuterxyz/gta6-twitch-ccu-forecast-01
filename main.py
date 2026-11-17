"""gta6-twitch-ccu-forecast-01 entry: scale-conditioned empirical-quantile forecast.

Model (fitted at run time from the case's own comparables, so it adapts to
whatever launches the case ships):

  For each comparable launch, take its release-day hourly curve (UTC day of
  its day_start) and the log-ratio L[h] = ln(viewers[h] / viewers[hour 1])
  for forecast hours h = 2..24. The shape of a launch day depends strongly
  on how big hour 1 already is: a huge hour-1 print means the game is
  already live in most regions and the day stays fairly flat (evening peak
  ~1.3x hour 1); a tiny hour-1 print means most regions have not unlocked
  yet and the count can explode 100x by the evening. So each comparable is
  weighted by a Gaussian kernel on log(hour-1 viewers) around the case's own
  hour-1 print (bandwidth 1.0 in log units), and the forecast quantiles are
  the kernel-weighted empirical quantiles of L[h], blended toward the
  kernel-weighted normal fit (which stabilises the 5%/95% tails). The blend
  is 30% normally but rises toward 65% when the kernel's effective sample
  size is small (under 15 launches) — which is exactly what happens if the
  case's hour-1 print is far above every comparable, as a huge GTA VI
  hour-1 print would be — because a 5% quantile read off six launches is
  mostly noise. One 3-hour smoothing pass across hours follows.

  q_tau(h) = p1 * exp(quantile_tau(L[h])), p1 = the case's hour-1 print.

  The "Grand Theft Auto VI category before release" comparable is not a
  launch and is excluded, as is any comparable with a missing/zero hour-1
  print or fewer than 20 observed hours on its release day.

  If the case ships no usable comparables, a hardcoded pooled prior (median
  shape and spread over 38 past launches) is used instead, so the program
  still emits a valid forecast. It never crashes on a malformed case: any
  failure to parse a field falls back to the prior, and the output is always
  23 rows, hours 2..24, five non-decreasing quantiles each, all in (0, 1e9].

  Deterministic: no randomness, no wall clock, no network, sorted file
  order, standard library only. `python3 -P -s /bot/main.py`, 8 s budget.

  Leave-one-out cross-validation on the 37 usable comparables in the kit
  (pinball loss on the log scale, mean over hours 2..24 x 5 levels):
  house persistence baseline 0.93 overall / 0.25 on big launches
  (hour-1 >= 50k); this model 0.39 overall / 0.21 on big launches.
  The house baseline's published practice loss is 0.387.
"""

import json
import math
import os
import sys
from pathlib import Path

# The grader's -P leaves /bot off sys.path: put it back first, so a module
# shipped beside this file imports.
sys.path.insert(0, str(Path(__file__).resolve().parent))

INPUT_DIR = Path(os.environ.get("MAKEFUN_INPUT_DIR", "/input"))
OUTPUT_DIR = Path(os.environ.get("MAKEFUN_OUTPUT_DIR", "/output"))

LEVELS = [0.05, 0.25, 0.5, 0.75, 0.95]
# Standard normal quantiles at the scored levels.
Z = [-1.6448536269514722, -0.6744897501960817, 0.0, 0.6744897501960817, 1.6448536269514722]

# Gaussian kernel bandwidth on log(hour-1 viewers).
BANDWIDTH = 1.0
# Blend of the empirical quantiles toward the kernel-weighted normal fit.
BLEND_NORMAL = 0.3
# Smoothing passes across neighbouring hours (3-point moving average).
SMOOTH_PASSES = 1

# Fallback prior (pooled over 38 past launches) used only when a case ships
# no usable comparables: (median log-ratio, log-spread) per forecast hour.
# Ratios: dip to ~0.75 early, evening peak ~2.4x; spread widens with horizon.
FALLBACK = {
    2: (-0.139, 0.299), 3: (-0.288, 0.715), 4: (-0.315, 0.760),
    5: (-0.288, 0.787), 6: (-0.236, 0.839), 7: (-0.223, 0.852),
    8: (-0.105, 0.887), 9: (-0.128, 1.056), 10: (-0.094, 1.406),
    11: (-0.073, 1.517), 12: (-0.020, 1.577), 13: (0.058, 1.683),
    14: (0.166, 1.834), 15: (0.419, 1.994), 16: (0.501, 2.111),
    17: (0.571, 2.160), 18: (0.668, 2.173), 19: (0.875, 2.168),
    20: (0.863, 2.170), 21: (0.846, 2.176), 22: (0.833, 2.158),
    23: (0.779, 2.124), 24: (0.713, 2.111),
}


def _norm_ppf(p: float) -> float:
    """Inverse standard normal CDF (Acklam's approximation), for levels
    outside the scored five. Accurate to ~1e-9; deterministic."""
    if p <= 0.0:
        return -8.0
    if p >= 1.0:
        return 8.0
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1.0 - 0.02425
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


def _z_for(levels):
    """z-scores for the case's levels (exact for the scored five)."""
    out = []
    for tau in levels:
        if tau in LEVELS:
            out.append(Z[LEVELS.index(tau)])
        else:
            out.append(_norm_ppf(tau))
    return out


def round2(x: float):
    """x rounded to 2 decimals, half up (as the house baseline rounds)."""
    scaled = x * 100.0
    whole = math.floor(scaled)
    if scaled - whole >= 0.5:
        whole += 1
    value = whole / 100.0
    return int(value) if float(value).is_integer() else value


def _parse_ts(text):
    """ISO-8601 UTC timestamp -> epoch seconds (None on failure)."""
    try:
        from datetime import datetime, timezone
        return datetime.fromisoformat(text.replace("Z", "+00:00")).timestamp()
    except Exception:
        return None


def _day_curve(comp):
    """The comparable's release-day hourly viewers: 24 slots (None=missing)."""
    day_start = _parse_ts(comp.get("day_start", ""))
    if day_start is None:
        return None
    curve = [None] * 24
    hourly = comp.get("hourly") or []
    for pt in hourly:
        if not isinstance(pt, dict):
            continue
        t = _parse_ts(pt.get("from", ""))
        v = pt.get("viewers")
        if t is None or v is None:
            continue
        h = int((t - day_start) // 3600)
        if 0 <= h < 24:
            try:
                curve[h] = float(v)
            except (TypeError, ValueError):
                pass
    return curve


def _usable_comparables(case):
    """(p1, curve) pairs usable for fitting the shape."""
    out = []
    for comp in case.get("comparables") or []:
        if not isinstance(comp, dict):
            continue
        unlock = str(comp.get("unlock", ""))
        if unlock.startswith("Not a launch"):
            continue
        if "before release" in str(comp.get("name", "")):
            continue
        curve = _day_curve(comp)
        if curve is None:
            continue
        p1 = curve[0]
        if p1 is None or p1 < 1:
            continue
        if sum(1 for v in curve if v is not None) < 20:
            continue
        out.append((float(p1), curve))
    return out


def _wquantile(pairs, tau):
    """Weighted quantile with linear interpolation between order statistics."""
    pairs = sorted(pairs)
    total = sum(w for _, w in pairs)
    if total <= 0:
        return None
    cum = 0.0
    pts = []
    for value, w in pairs:
        cum += w
        pts.append((cum - 0.5 * w, value))
    target = tau * total
    if target <= pts[0][0]:
        return pts[0][1]
    if target >= pts[-1][0]:
        return pts[-1][1]
    for i in range(1, len(pts)):
        if pts[i][0] >= target:
            (c0, v0), (c1, v1) = pts[i - 1], pts[i]
            if c1 == c0:
                return v1
            return v0 + (target - c0) / (c1 - c0) * (v1 - v0)
    return pts[-1][1]


def _fit_log_quantiles(comps, p1, levels, z):
    """Per-hour log-quantiles of L[h] = ln(viewers[h] / viewers[hour 1])."""
    log_p1 = math.log(max(p1, 1.0))
    out = {}
    for h in range(2, 25):
        pairs = []
        for cp1, curve in comps:
            v = curve[h - 1]
            if v is None or v <= 0:
                continue
            w = math.exp(-0.5 * ((math.log(cp1) - log_p1) / BANDWIDTH) ** 2)
            pairs.append((math.log(v / cp1), w))
        if not pairs:
            mu, sd = FALLBACK[h]
            out[h] = [mu + zz * sd for zz in z]
            continue
        emp = [_wquantile(pairs, tau) for tau in levels]
        total = sum(w for _, w in pairs)
        mu = sum(v * w for v, w in pairs) / total
        var = sum(w * (v - mu) ** 2 for v, w in pairs) / total
        sd = math.sqrt(max(var, 1e-6))
        norm = [mu + zz * sd for zz in z]
        # Effective sample size of the kernel weights: when the case's
        # hour-1 print sits far from every comparable, few launches carry
        # weight and the empirical tails are noisy, so lean harder on the
        # normal fit (blend 0.30 -> up to 0.65 as ESS falls below 15).
        ess = total * total / sum(w * w for _, w in pairs)
        blend = BLEND_NORMAL
        if ess < 15.0:
            blend = min(0.65, BLEND_NORMAL + 0.35 * (1.0 - ess / 15.0))
        out[h] = [(1.0 - blend) * e + blend * n
                  for e, n in zip(emp, norm)]
    for _ in range(SMOOTH_PASSES):
        smoothed = {}
        for h in range(2, 25):
            hs = [hh for hh in (h - 1, h, h + 1) if 2 <= hh <= 24]
            smoothed[h] = [sum(out[hh][i] for hh in hs) / len(hs)
                           for i in range(len(levels))]
        out = smoothed
    return out


def forecast_case(case, name):
    levels = case.get("levels") or LEVELS
    levels = [float(t) for t in levels]
    z = _z_for(levels)
    p1 = 1.0
    for row in case.get("known") or []:
        if isinstance(row, dict) and row.get("hour") == 1:
            try:
                p1 = max(1.0, float(row.get("viewers", 1) or 1))
            except (TypeError, ValueError):
                p1 = 1.0
            break
    comps = _usable_comparables(case)
    if comps:
        log_q = _fit_log_quantiles(comps, p1, levels, z)
    else:
        log_q = {h: [FALLBACK[h][0] + zz * FALLBACK[h][1] for zz in z]
                 for h in range(2, 25)}
    hours = []
    for row in case.get("forecast") or []:
        h = int(row.get("hour", 0))
        if h < 2 or h > 24:
            continue
        qs = []
        prev = 0.0
        for x in log_q[h]:
            q = p1 * math.exp(x)
            q = min(max(q, 1.0), 1e9)
            q = max(q, prev)  # keep non-decreasing after rounding/clamping
            qs.append(round2(q))
            prev = qs[-1]
        hours.append({"hour": h, "q": qs})
    if len(hours) != 23 or [r["hour"] for r in hours] != list(range(2, 25)):
        # The forecast list was unexpected; rebuild from hours 2..24 directly.
        hours = []
        for h in range(2, 25):
            qs, prev = [], 0.0
            for x in log_q[h]:
                q = min(max(p1 * math.exp(x), 1.0), 1e9)
                q = max(q, prev)
                qs.append(round2(q))
                prev = qs[-1]
            hours.append({"hour": h, "q": qs})
    return {"hours": hours}


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(INPUT_DIR.glob("*.in")):
        try:
            case = json.loads(path.read_text(encoding="utf-8"))
            forecast = forecast_case(case, path.name)
        except Exception:
            # Never crash a case: emit the fallback prior around a flat print.
            levels, z = LEVELS, Z
            forecast = {"hours": [{"hour": h,
                                   "q": [round2(min(max(math.exp(FALLBACK[h][0] + zz * FALLBACK[h][1]), 1.0), 1e9)) for zz in z]}
                                  for h in range(2, 25)]}
        (OUTPUT_DIR / f"{path.stem}.out").write_text(
            json.dumps(forecast, separators=(",", ":")) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
