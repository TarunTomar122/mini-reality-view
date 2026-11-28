# Neon — Exit-6 adjudicator re-check (28 Nov 2026; commit c5ad60b)
Target: scripts/wan_gate_adjudicate.py (sha256 fa6af9ae37619e796916cf641dcc4fda86675729aaed39a410024629cbe07fde).
1. Fixes 1-3 verified: tokens_only counts/F1 finite + count-consistent and allow_nan=False on append; loss_pass is False, f1_pass is True, roundtrip_positive is True; isint rejects bool in prior_not_reached/counts/eval_history/log.jsonl and dict guards refuse non-dict overfit.json/counts/car_f1_controls without crashing.
2. Tests & repros: python3 scripts/wan_tests.py -> WAN_TESTS FAILS=0; all 13 Fix 1-3 repros accept/crash on HEAD~1 and cleanly refuse (rc=1, 0 bytes) on HEAD; 15 extra cases (null/scalar overfit.json, 0/1/None flags, missing/neg tokens_only, bool log update, append_once NaN safety, double-2000 tolerance) pass.
3. Diff & freeze: git show HEAD changes only the three fixes (signed constants, wiring/fitting/registry rules, and appended record schema unchanged); all 13 council/FREEZE-s1.md sha256s match.
VERIFIED: yes
