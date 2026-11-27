"""CPU fixtures for mrv/noise.py (A3 vpriv). Exits 1 on failure.  python scripts/test_noise.py"""
import sys, torch
sys.path.insert(0, "."); from mrv.noise import build_eps
fails = 0
def check(name, ok, msg=""):
    global fails; fails += not ok; print(("PASS " if ok else "FAIL ") + name, msg)
g = torch.Generator().manual_seed(0)
C, F, rho = 2, 16, 0.95
ec = torch.randn(C, 1, 4, 32, 64, generator=g); eo = torch.randn(C, F, 4, 32, 64, generator=g)
old = (rho ** 0.5 * ec + (1 - rho) ** 0.5 * eo).reshape(C * F, 4, 32, 64)
check("private 0 == AGREED-r1 formula (bit-identical)", torch.equal(build_eps(ec, eo, rho, 0, 7), old))
def tick_corr(e):  # mean correlation between consecutive ticks of one clip
    e = e.reshape(F, -1); return torch.stack([torch.corrcoef(torch.stack([e[i], e[i + 1]]))[0, 1] for i in range(F - 1)]).mean().item()
e2 = build_eps(ec, eo, rho, 2, 0).reshape(C, F, 4, 32, 64)
c0, c1 = tick_corr(e2[0]), tick_corr(e2[1])
check("private 2: both clips independent per tick", abs(c0) < 0.02 and abs(c1) < 0.02, f"corr {c0:.4f} {c1:.4f}")
check("private 2: unit variance", abs(e2.std().item() - 1) < 0.01, f"std {e2.std().item():.4f}")
for step in (0, 1, 2, 3):
    e1 = build_eps(ec, eo, rho, 1, step).reshape(C, F, 4, 32, 64)
    p, s = step % C, 1 - step % C
    cp, cs = tick_corr(e1[p]), tick_corr(e1[s])
    check(f"private 1, step {step}: slot {p} private, slot {s} shared", abs(cp) < 0.02 and abs(cs - rho) < 0.02, f"corr private {cp:.4f} shared {cs:.4f}")
e0 = build_eps(ec, eo, rho, 0, 0).reshape(C, F, 4, 32, 64)
check("private 0: both clips shared at rho", all(abs(tick_corr(e0[i]) - rho) < 0.02 for i in range(C)))
print("NOISE_TESTS", "ALL PASS" if not fails else f"{fails} FAIL"); sys.exit(1 if fails else 0)
