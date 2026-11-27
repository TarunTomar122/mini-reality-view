"""CPU fixtures for scripts/rho_rule.py (Astra review 20:30 cases + the plain paths). Exits 1 on any failure.
  python scripts/test_rho_rule.py
"""
import sys
from rho_rule import decide, f1_agrees

A = 0.4456


def row(idr, wr, f1=0.45, adm=True):
    return {"admissible": adm, "ID_ratio": idr, "W_ratio": wr, "meets": idr <= 1.20 and wr <= 1.20, "f1": f1, "f1_A": A}


MISS = {"admissible": False}
P05 = row(1.496, 1.270, 0.4597)  # the real rho 0.5 result
cases = [
    ("0.8 meets -> rule 0.8", {"0.5": P05, "0.8": row(1.15, 1.10)}, None, ("rule", 0.8)),
    ("0.5 meets -> rule 0.5", {"0.5": row(1.1, 1.1), "0.8": MISS}, None, ("rule", 0.5)),
    ("0.8 missing -> wait", {"0.5": P05, "0.8": MISS}, None, (None, None)),
    ("0.9 missing, 0.95 passes -> wait, never skip a rung", {"0.5": P05, "0.8": row(1.3, 1.2), "0.9": MISS, "0.95": row(1.1, 1.0)}, 10, (None, None)),
    ("0.9 meets but 0.95 missing -> wait for the complete panel", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.18, 1.1), "0.95": MISS}, 10, (None, None)),
    ("0.9 meets, 0.95 missing, deadline -> override only", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.18, 1.1), "0.95": MISS}, 90.0, ("override", 0.8)),
    ("0.9 meets, complete panel -> fallback 0.9", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.18, 1.1), "0.95": row(1.12, 1.05)}, 10, ("fallback", 0.9)),
    ("0.9 meets but off@0.8 F1 below floor -> council", {"0.5": P05, "0.8": row(1.3, 1.25, 0.300), "0.9": row(1.18, 1.1), "0.95": row(1.12, 1.05)}, 10, (None, None)),
    ("F1 just below floor (full precision) -> council", {"0.5": P05, "0.8": row(1.3, 1.25, A - 0.03 - 1e-6), "0.9": row(1.25, 1.22), "0.95": row(1.22, 1.21)}, 20, (None, None)),
    ("Astra case: candidate 0.41556 vs A 0.44564 -> council", {"0.5": P05, "0.8": dict(row(1.3, 1.25), f1=0.41556, f1_A=0.44564), "0.9": row(1.25, 1.22), "0.95": row(1.22, 1.21)}, 20, (None, None)),
    ("0.9 fails, 0.95 meets -> fallback 0.95", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.25, 1.1), "0.95": row(1.15, 1.05)}, 10, ("fallback", 0.95)),
    ("both higher missing, 10 min -> wait (no early override)", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": MISS, "0.95": MISS}, 10, (None, None)),
    ("deadline 89.99 min -> still wait", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": MISS, "0.95": MISS}, 89.99, (None, None)),
    ("deadline 90.0 min -> override 0.8", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": MISS, "0.95": MISS}, 90.0, ("override", 0.8)),
    ("both higher admissible failures -> override 0.8", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.25, 1.22), "0.95": row(1.22, 1.21)}, 20, ("override", 0.8)),
    ("0.95 meets but its F1 below floor -> council", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.25, 1.22), "0.95": row(1.1, 1.0, 0.300)}, 20, (None, None)),
    ("override but 0.8 F1 below floor -> council", {"0.5": P05, "0.8": row(1.3, 1.25, 0.300), "0.9": row(1.25, 1.22), "0.95": row(1.22, 1.21)}, 20, (None, None)),
    ("F1 exactly at floor (A - 0.03) passes", {"0.5": P05, "0.8": row(1.3, 1.25, A - 0.03), "0.9": row(1.25, 1.22), "0.95": row(1.22, 1.21)}, 20, ("override", 0.8)),
    ("bug: 0.8 worse than 0.5 on both -> council", {"0.5": P05, "0.8": row(1.6, 1.3), "0.9": row(1.25, 1.22), "0.95": row(1.22, 1.21)}, 20, (None, None)),
    ("bug: 0.95 worse than 0.9 on both -> council", {"0.5": P05, "0.8": row(1.3, 1.25), "0.9": row(1.21, 1.21), "0.95": row(1.25, 1.24)}, 20, (None, None)),
    ("0.5 inadmissible -> no rule", {"0.5": MISS, "0.8": row(1.1, 1.1)}, None, (None, None)),
]
agree_cases = [  # (raw, signed, expected) -- Astra a2r2: exact 4-decimal agreement, finite only
    (0.41561, 0.4156, True), (0.41561, 0.4155, False), (0.44564, 0.4456, True), (0.44564, 0.4457, False),
    (float("nan"), 0.4456, False), (0.4456, float("nan"), False), (float("inf"), 0.4456, False), (None, 0.4456, False)]
fails = 0
for raw, signed, want in agree_cases:
    got = f1_agrees(raw, signed); ok = got == want; fails += not ok
    print(("PASS" if ok else "FAIL"), f"f1_agrees({raw}, {signed}) -> {got} (want {want})")
for name, rows, el, want in cases:
    mode, rho, note = decide(rows, el)
    ok = (mode, rho) == want
    fails += not ok
    print(("PASS" if ok else "FAIL"), name, "->", mode, rho, "|", note)
print("RHO_RULE_TESTS", "ALL PASS" if not fails else f"{fails} FAIL")
sys.exit(1 if fails else 0)
