# Fixed pooled-18 rendered-check frame list for car map R (PLAN-carmap v4): 6 dev drives (degraded boxes) + 12 reserved
# calibration drives (raw boxes); 4 evenly spaced keyed keyframes per drive (all of them if fewer), seeds 0/1;
# map-only swap donor = the drive half the list away, frame at the same relative position among the donor's frames.
import json, glob, os, numpy as np, torch
L = json.load(open("data/temporal_lists.json")); dev = L["dev"]; res = open("tmp/carmap_dev12.txt").read().split()
drives = dev + res
def frames(sc):
    kk = sorted(glob.glob(f"data/sim/keys_tpv/{sc}/*.png")); nf = min(4, len(kk))
    out = []
    for i in np.linspace(0, len(kk) - 1, nf).round().astype(int):
        b = os.path.basename(kk[i])[:-4]; out.append((int(b[:3]), b[4:]))
    return out
F = {sc: frames(sc) for sc in drives}
for sc in drives:
    e = torch.load(f"data/enc_full/{sc}.pt")
    for k, t in F[sc]: assert e["sample_tokens"][k] == t, (sc, k)
items = []
for di, sc in enumerate(drives):
    dn = drives[(di + len(drives) // 2) % len(drives)]
    for q, (k, t) in enumerate(F[sc]):
        dq = round(q / max(1, len(F[sc]) - 1) * (len(F[dn]) - 1)) if len(F[sc]) > 1 else 0
        dk, dt = F[dn][dq]
        items.append({"scene": sc, "k": k, "token": t, "donor": dn, "donor_k": dk, "donor_token": dt, "group": "dev6" if sc in dev else "reserved12"})
json.dump({"frames": items, "seeds": [0, 1], "drives": len(drives), "rule": "see header of council/carmap_dev18_frames.json generator (tmp/mk18.py)"}, open("tmp/carmap_dev18_frames.json", "w"), indent=1)
print("drives", len(drives), "frames", len(items), {sc: len(F[sc]) for sc in drives if len(F[sc]) != 4})
