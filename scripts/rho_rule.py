"""Pure decision logic for the rho choice (council/AGREED-causal-r1.md + amendment A2), separated from file I/O so it can be
tested on CPU fixtures (scripts/test_rho_rule.py).

rows: {"0.5" | "0.8" | "0.9" | "0.95": row}; a row is admissible evidence or not; admissible rows carry ID_ratio, W_ratio,
meets (both <= 1.20), f1 (off car F1) and f1_A (signed A@20k car F1).
elapsed_min: minutes since the FIRST recorded admissible 0.8 failure (None if 0.8 has not failed admissibly).
Returns (mode, rho, note): mode in {"rule", "fallback", "override", None}; None = no launch (wait or council, see note).

Rules
- AGREED r1: 0.5 then 0.8; the first admissible row that meets is chosen ("rule"); an inadmissible row stops the rule.
- A2 step 2 (only after an admissible 0.8 failure): the lowest of 0.9, 0.95 that meets, but a rung is only reachable when every
  lower rung is admissible and failed; a missing rung = wait (unless the 90-min deadline passed).
- Bug guard: any admissible higher rung worse than an admissible lower rung on BOTH ratios -> council.
- F1 floor (reconciled Astra / Fable): the off car F1 at the rho that would TRAIN must be >= A - 0.03 (fallback and override).
- A2 step 3 override (rho 0.8): when 0.9 and 0.95 are both admissible failures, or the 90-min deadline has passed; requires 0.5
  and 0.8 admissible, 0.8 not worse than 0.5 on both ratios, no bug, F1 floor at 0.8.
"""
RUNGS = ("0.5", "0.8", "0.9", "0.95")
DEADLINE_MIN = 90.0


def _adm(rows, r):
    return bool(rows.get(r, {}).get("admissible"))


def _both_worse(rows, hi, lo):
    return _adm(rows, hi) and _adm(rows, lo) and rows[hi]["ID_ratio"] > rows[lo]["ID_ratio"] and rows[hi]["W_ratio"] > rows[lo]["W_ratio"]


def _f1_ok(rows, r):
    return rows[r]["f1"] >= rows[r]["f1_A"] - 0.03


def decide(rows, elapsed_min=None):
    for r in ("0.5", "0.8"):
        if not _adm(rows, r):
            return None, None, f"rho {r} evidence not admissible: the AGREED rule cannot be applied (wait / fix and rescore)"
        if rows[r]["meets"]:
            return "rule", float(r), f"rho {r} meets the rule"
    # 0.5 and 0.8 are admissible failures from here on
    bug = [f"{hi} worse than {lo} on both" for i, lo in enumerate(RUNGS) for hi in RUNGS[i + 1:] if _both_worse(rows, hi, lo)]
    if bug:
        return None, None, "bug guard: " + "; ".join(bug) + " -> council"
    timed_out = elapsed_min is not None and elapsed_min >= DEADLINE_MIN
    for r in ("0.9", "0.95"):
        if not _adm(rows, r):
            if timed_out:
                break  # deadline passed with an incomplete ladder -> step 3
            return None, None, f"waiting for admissible rho {r} evidence ({elapsed_min} of {DEADLINE_MIN:.0f} min)"
        if rows[r]["meets"]:
            if not _f1_ok(rows, r):
                return None, None, f"rho {r} meets the ratios but off car F1 {rows[r]['f1']:.4f} < A - 0.03 -> council"
            return "fallback", float(r), f"rho {r} is the lowest rung meeting the rule"
    # step 3: guarded override at 0.8
    if not _f1_ok(rows, "0.8"):
        return None, None, f"override refused: off@0.8 car F1 {rows['0.8']['f1']:.4f} < A - 0.03 -> council"
    return "override", 0.8, "no rung meets" + (" (90-min deadline passed)" if timed_out else "") + ": guarded override at 0.8"
