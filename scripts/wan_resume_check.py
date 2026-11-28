"""Astra items 11 / S1b 10: interrupted + resumed training must equal uninterrupted training (slurm/wan_teacher.sh MODE=resume_test
runs 16 updates straight vs 8 + resume 8). Compares two trainer checkpoints: same step, same chained data hash (every batch's
picks / noise / sigma / dropout), same init sha, same optimizer step counts, and EVERY trainable tensor within its OWN tolerance:
max |a - b| <= --rel x max |a| (that tensor's scale) + --abs, so a diverging small tensor (a gate, a LoRA-B) cannot hide behind
large weights; Adam moments are checked the same way. Bitwise equality is reported (GPU attention backward uses atomics, so it is
not required). Exit 0 = equivalent.
  python scripts/wan_resume_check.py runs/wan_ctest_resA/ckpt.pt runs/wan_ctest_resB/ckpt.pt
"""
import argparse, json, sys
import torch

p = argparse.ArgumentParser(); p.add_argument("a"); p.add_argument("b")
p.add_argument("--rel", type=float, default=1e-3); p.add_argument("--abs", type=float, default=1e-6)
a = p.parse_args()
A = torch.load(a.a, map_location="cpu", weights_only=False); B = torch.load(a.b, map_location="cpu", weights_only=False)
res = {"step": (A["step"], B["step"]), "chain_equal": A["pick_chain"] == B["pick_chain"], "init_equal": A.get("init_sha") == B.get("init_sha")}
sa = sorted({int(v["step"]) for v in A["opt"]["state"].values()}); sb = sorted({int(v["step"]) for v in B["opt"]["state"].values()})
res["adam_steps"] = (sa, sb)


def compare(da, db, label):
    bad, worst, bitwise = [], [], True
    if set(da) != set(db):
        return [f"{label}: key sets differ"], [], False
    for k in sorted(da):
        x, y = da[k].float(), db[k].float()
        if x.shape != y.shape:
            bad.append(f"{label}:{k} shape"); continue
        d = (x - y).abs().max().item() if x.numel() else 0.0
        tol = a.rel * (x.abs().max().item() if x.numel() else 0.0) + a.abs
        bitwise &= d == 0.0
        worst.append((d / tol if tol else float("inf"), k, d, tol))
        if not d <= tol:
            bad.append(f"{label}:{k} max|diff| {d:.3g} > {tol:.3g}")
    worst.sort(reverse=True)
    return bad, worst[:8], bitwise


bad_w, worst_w, bit_w = compare(A["trainable"], B["trainable"], "weight")
ma = {f"{i}.{n}": v for i, s_ in A["opt"]["state"].items() for n, v in s_.items() if torch.is_tensor(v) and v.numel() > 1}
mb = {f"{i}.{n}": v for i, s_ in B["opt"]["state"].items() for n, v in s_.items() if torch.is_tensor(v) and v.numel() > 1}
bad_m, worst_m, bit_m = compare(ma, mb, "adam")
gates = [k for k in A["trainable"] if k.endswith(".attn2.gate")]
res.update(tensors=len(A["trainable"]), gates_checked=len(gates), bitwise_weights=bit_w, bitwise_adam=bit_m,
           failing=(bad_w + bad_m)[:20], worst_weight_ratio=[(round(r, 3), k) for r, k, _, _ in worst_w],
           worst_adam_ratio=[(round(r, 3), k) for r, k, _, _ in worst_m], tol={"rel": a.rel, "abs": a.abs})
res["PASS"] = bool(A["step"] == B["step"] and res["chain_equal"] and res["init_equal"] and sa == sb and not bad_w and not bad_m)
print("RESUME_CHECK", json.dumps(res))
sys.exit(0 if res["PASS"] else 1)
