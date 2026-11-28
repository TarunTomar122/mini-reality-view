"""CPU preflight of the six dev camera-swap assignments (Astra / Neon 08:30), run before any panel (scripts/wan_panel.py runs it
first and stops on failure). For every dev receiver, mrv.wan_plan.plan_donors picks a donor:
  full mode   : donor at least as long as the receiver (the conventional pairing if it qualifies, else the closest in length);
  shared mode : only if no full-length donor qualifies: the receiver is cut to the longest qualifying donor's length n and the
                swap is compared with an unswapped receiver run of the same n.
Each assignment must have: no truncation of its reference length, identical segments / windows / noise keys to the unswapped
reference run, common placement frames after the swap (receiver AND donor tick % 5 == 0, accepted, both keys present), and
structurally lag-eligible 5-tick windows (consecutive accepted donor ticks with keys, no restart inside). Writes --out json
({"donors": {receiver: {"donor", "mode", "n"}}, "checks", "ok"}); exit 1 if any receiver has no qualifying donor.
  python scripts/wan_swap_preflight.py --out runs/wroll/swap_donors.json
"""
import argparse, json, os, sys
import torch
from mrv import wan_plan as WP

p = argparse.ArgumentParser()
p.add_argument("--lists", default="data/temporal_lists.json"); p.add_argument("--enc", default="data/enc_tpv10")
p.add_argument("--keys", default="data/sim/keys_tpv10"); p.add_argument("--window_latents", type=int, default=5)
p.add_argument("--swap_at", type=int, default=50); p.add_argument("--out", required=True)
a = p.parse_args()
L = json.load(open(a.lists))
dev = WP.check_split_membership(list(L["dev"]), L, allowed=("dev",))  # list order = the conventional pairing order
drives = {}
for d in dev:
    E = torch.load(f"{a.enc}/{d}.pt", map_location="cpu", weights_only=False, mmap=True)
    drives[d] = {"ticks": [int(t) for t in E["ticks"].tolist()], "ok": [bool(x) for x in E["ok"].tolist()], "hard": [bool(x) for x in E["hard"].tolist()]}
has_key = lambda sc, t: os.path.exists(f"{a.keys}/{sc}/t{int(t):04d}.jpg")
plan = WP.plan_donors(drives, a.window_latents, a.swap_at, has_key)
plan.update(lengths={d: len(v["ticks"]) for d, v in drives.items()}, window_latents=a.window_latents, swap_at=a.swap_at)
os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
json.dump(plan, open(a.out, "w"), indent=1)
print("SWAP_PREFLIGHT", "PASS" if plan["ok"] else "FAIL", json.dumps(plan["donors"]))
sys.exit(0 if plan["ok"] else 1)
