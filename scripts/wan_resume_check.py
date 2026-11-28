"""Astra item 11: interrupted + resumed training must equal uninterrupted training (slurm/wan_teacher.sh MODE=resume_test runs both).
Compares two trainer checkpoints: same step, same chained data hash (every batch's picks / noise / sigma / dropout), same
optimizer step counts, and the trainable weights: bit-identical, or (GPU atomics in attention backward are not bitwise
deterministic) max |diff| <= 1e-3 x max |w| — reported either way. Exit 0 = equivalent.
  python scripts/wan_resume_check.py runs/wan_ctest_resA/ckpt.pt runs/wan_ctest_resB/ckpt.pt
"""
import json, sys
import torch

A = torch.load(sys.argv[1], map_location="cpu", weights_only=False); B = torch.load(sys.argv[2], map_location="cpu", weights_only=False)
res = {"step": (A["step"], B["step"]), "chain_equal": A["pick_chain"] == B["pick_chain"], "init_equal": A.get("init_sha") == B.get("init_sha")}
sa = sorted({int(v["step"]) for v in A["opt"]["state"].values()}); sb = sorted({int(v["step"]) for v in B["opt"]["state"].values()})
res["adam_steps"] = (sa, sb)
keys_ok = set(A["trainable"]) == set(B["trainable"])
mx = max((A["trainable"][k] - B["trainable"][k]).abs().max().item() for k in A["trainable"]) if keys_ok else float("inf")
scale = max(A["trainable"][k].abs().max().item() for k in A["trainable"])
res.update(bitwise=bool(keys_ok and mx == 0.0), max_abs_diff=mx, max_abs_w=scale)
res["PASS"] = bool(A["step"] == B["step"] and res["chain_equal"] and res["init_equal"] and sa == sb and keys_ok and mx <= 1e-3 * scale)
print("RESUME_CHECK", json.dumps(res))
sys.exit(0 if res["PASS"] else 1)
