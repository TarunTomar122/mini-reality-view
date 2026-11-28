"""ONE verifier of an A5 cache_check record, shared by scripts/vclean_gate.py (passed state), scripts/train_clean.py (passed
and pending states, before the first cache read and on every resume) and scripts/cache_check.py (self-check of its output)
(council/codereview-vmix-astra.md, codereview-vmix2-astra.md, codereview-vmix3-astra.md).

INDEPENDENT PINS (council/vmix_pins.json, part of the reviewed code set): the expected parity-reference artifact (path, drive and
sha256 of the unchanged causal_rollout render of scene-0005, made outside the candidate record) and the six known-good full-file
digests of the signed seed-0 dev renders. The verifier refuses while the parity pin is PENDING and never takes either from the
candidate record. Tests pass a separate test-only pins file explicitly (bind["pins"] != the production path).

Every file is read ONCE into memory; its digest is computed on those bytes and it is parsed from the same bytes (json.loads /
torch.load(io.BytesIO(buf))), never reopened.

A "bind" fixes what a record must prove, independently of the record: the trainer's ACTUAL A@20k file digest, the cache dir,
the input paths (lists / enc / rays), the exact train and dev drive lists, the six signed seed-0 dev reference digests, and the
reviewed sha256 of every file of the cache chain (builder, checker, matched rollout + its dependencies, this verifier).
verify(path, bind, data) accepts a record only if
  * it is a passing cache_check record (no failures) of that cache dir, checkpoint and inputs, written by the reviewed checker,
    and the reviewed files are the ones on disk now;
  * its cache was rendered by the reviewed builder (roll_code) at rho 1.0, seed 0, unmerged, and covers EXACTLY the 144 train
    drives; every cache file matches its recorded digest, is fp16 (n, 4, 32, 64) with ticks aligned to the trainer's encoding,
    carries its signed fields, and its ENCODING digest equals both the record's and the digest of the encoding bytes the caller
    actually loaded (data["enc_sha"]); its ray digest equals the caller's rays;
  * parity is RE-MEASURED: the PINNED reference artifact (the unchanged causal_rollout's output for the pinned parity drive;
    path and digest from the pins file, never from the record) is re-loaded and compared with the cached drive: whole drive, ticks / source ticks / ok / hard / restart / amp
    equal, same inputs, A@20k memory off at rho 1.0, seed-0 noise keys, max |diff| < 0.01;
  * the six dev references are exactly the PINNED known-good files (full-file digests), A@20k independent renders, and seed 0
    is MEASURED (the files carry no seed field): the recorded seed-check artifact (causal_rollout --seeds 0 --max_ticks 5, seed-0
    noise keys) is the COMPLETE five-tick prefix with matching hard flags and matches on every non-hard tick (< 0.01);
  * every train and dev cache-to-key MSE is present and finite.
resume_identity() is the trainer's resume rule (gate approval and cache pin preserved across resumes).
"""
import hashlib, io, json, math, os

PINS = "council/vmix_pins.json"
REVIEWED = ["scripts/build_cache.py", "scripts/cache_check.py", "scripts/causal_rollout.py", "mrv/causal_video.py", "mrv/sd.py", "mrv/cache_verify.py", PINS]
A_PATH = "runs/longA/ckpt_20000.pt"
DEV_REF = "runs/troll/A20k_indep_dev"


def read(f):
    with open(f, "rb") as fh:
        return fh.read()


def hb(buf):
    return hashlib.sha256(buf).hexdigest()


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def tload(buf):
    import torch
    return torch.load(io.BytesIO(buf), map_location="cpu")


def file_code(files=REVIEWED):
    return {f: sha(f) for f in files}


def builder_roll_code():  # == scripts/build_cache.py ROLL_CODE
    return hashlib.sha256(b"".join(read(f) for f in ["scripts/build_cache.py", "mrv/sd.py"])).hexdigest()[:12]


def rollout_roll_code():  # == scripts/causal_rollout.py ROLL_CODE
    return hashlib.sha256(b"".join(read(f) for f in ["scripts/causal_rollout.py", "mrv/causal_video.py", "mrv/sd.py"])).hexdigest()[:12]


def ray_sha(t):  # == build_cache / causal_rollout RAY_SHA
    return hashlib.sha256(t.float().numpy().tobytes()).hexdigest()[:16]


def load_pins(path):
    b = read(path)
    return json.loads(b), hb(b)


def make_bind(ckpt, cache, lists, enc, rays, code, pins=PINS, ckpt_sha=None):
    """ckpt_sha: the caller's digest of the A bytes it parsed (else hashed here)"""
    L = json.loads(read(lists)); P, psha = load_pins(pins)
    return {"ckpt_sha": ckpt_sha or sha(ckpt)[:16], "cache": cache, "lists": lists, "enc": enc, "rays": rays, "train": sorted(L["train"]), "dev": list(L["dev"]),
            "pins": pins, "pins_sha256": psha, "dev_ref": P["dev_ref"], "code_sha256": {f: code[f] for f in REVIEWED}}


fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class Bad(Exception):
    pass


def _need(c, m):
    if not c:
        raise Bad(m)


def _maxdiff(u, v):
    return (u.float() - v.float()).abs().max().item()


def verify(path, bind, data):
    """-> (ok, reason, record sha256, {scene: fp16 latents} or None).
    data = {"ticks": {scene: tensor}, "rays": {scene: tensor}, "enc_sha": {scene: sha256(encoding bytes)[:16]}} from the caller's loads"""
    import torch
    try:
        _need(len(bind["train"]) == 144 and len(bind["dev"]) == 6, f"bind has {len(bind['train'])} train / {len(bind['dev'])} dev drives (A5: 144 / 6)")
        _need(path and os.path.isfile(path), f"no cache record at {path or '(none given)'}")
        P, psha = load_pins(bind["pins"])  # the independent pins (read once)
        _need(psha == bind["pins_sha256"], f"pins file {bind['pins']} changed since the bind")
        if bind["pins"] == PINS:
            _need(psha == bind["code_sha256"][PINS], "pins file is not the reviewed one")
        pin_par = P.get("parity_ref_sha256", "")
        _need(isinstance(pin_par, str) and len(pin_par) == 64 and all(ch in "0123456789abcdef" for ch in pin_par),
              f"parity reference pin is {pin_par!r} (PENDING / malformed): no cache use until it is pinned")
        _need(sorted(P.get("dev_ref_sha256", {})) == sorted(bind["dev"]) and P.get("dev_ref") == bind["dev_ref"], "pins do not cover the six dev drives")
        rbuf = read(path); rsha = hb(rbuf); rec = json.loads(rbuf)
        _need(rec.get("kind") == "cache_check" and rec.get("pass") is True and rec.get("failures") == [], f"not a passing cache_check record (failures {rec.get('failures')})")
        for k in ("cache", "ckpt_sha", "lists", "enc", "rays", "dev_ref"):
            _need(rec.get(k) == bind[k], f"record {k} {rec.get(k)!r} != bound {bind[k]!r}")
        cur = file_code()
        for f in REVIEWED:
            _need(cur[f] == bind["code_sha256"][f], f"{f} on disk is not the reviewed file")
            _need(rec.get("code_sha256", {}).get(f) == bind["code_sha256"][f], f"record was not produced with the reviewed {f}")
        _need(rec.get("roll_code") == builder_roll_code(), "cache not rendered by the reviewed builder (roll_code)")
        b = rec.get("build", {})
        _need(b.get("merge") is False and isinstance(b.get("batch"), int) and b.get("amp"), f"build settings {b}")
        _need(sorted(rec.get("files", {})) == bind["train"] and rec.get("n_files") == 144, "record does not cover exactly the 144 train drives")
        _need(sorted(rec.get("enc_sha", {})) == bind["train"], "record lacks per-drive encoding digests")
        m = rec.get("mse", {})
        _need(sorted(m.get("train_per_drive", {})) == bind["train"] and all(fin(v) for v in m["train_per_drive"].values()), "train MSE missing / non-finite")
        _need(sorted(m.get("dev_per_drive", {})) == sorted(bind["dev"]) and all(fin(v) for v in m["dev_per_drive"].values()), "dev MSE missing / non-finite")
        _need(fin(m.get("train_mean")) and fin(m.get("dev_mean")), "MSE means non-finite")
        # ---- every cache file: digest of the bytes parsed, shape / dtype / ticks, signed fields, encoding + ray digests ----
        lat, files = {}, {}
        for sc in bind["train"]:
            f = f"{bind['cache']}/{sc}_s0.pt"
            _need(os.path.isfile(f), f"cache file {f} missing")
            buf = read(f)
            _need(hb(buf) == rec["files"][sc], f"cache file {f}: digest differs from the record")
            c = tload(buf); del buf
            x, tk = c["lat"], data["ticks"][sc]
            _need(x.dtype == torch.float16 and tuple(x.shape) == (len(tk), 4, 32, 64) and torch.equal(c["ticks"], tk), f"{f}: shape / dtype / ticks")
            _need(c.get("ckpt_sha") == bind["ckpt_sha"] and c.get("roll_code") == rec["roll_code"] and c.get("method") == "cache_off"
                  and c.get("rho") == 1.0 and c.get("seed") == 0 and c.get("merge") is False, f"{f}: signed fields")
            _need(c.get("enc_sha") == rec["enc_sha"][sc] == data["enc_sha"][sc], f"{f}: built from encoding {c.get('enc_sha')}, record {rec['enc_sha'][sc]}, "
                  f"trainer loaded {data['enc_sha'][sc]}")
            _need(c.get("ray_sha") == ray_sha(data["rays"][sc]), f"{f}: rays changed since the cache was built")
            lat[sc] = x; files[sc] = c
        # ---- parity re-measured against the INDEPENDENTLY pinned reference artifact ----
        pa = rec.get("parity", {}); drv = pa.get("drive")
        _need(drv == P.get("parity_drive") and pa.get("ref_file") == P.get("parity_ref") and pa.get("ref_file_sha256") == pin_par,
              f"record's parity reference ({drv}, {pa.get('ref_file')}) is not the pinned one ({P.get('parity_drive')}, {P.get('parity_ref')})")
        _need(drv in bind["train"] and os.path.isfile(P["parity_ref"]), f"pinned parity reference artifact {P.get('parity_ref')} missing")
        pbuf = read(P["parity_ref"])
        _need(hb(pbuf) == pin_par, "parity reference artifact differs from the pinned digest")
        xr, c = tload(pbuf), files[drv]; del pbuf
        _need(xr.get("ckpt_sha") == bind["ckpt_sha"] and xr.get("rho") == 1.0 and xr.get("method") == "causal_off" and xr.get("roll_code") == rollout_roll_code()
              and xr.get("salt_from", -1) == -1 and xr.get("start_tick", 0) == 0 and not xr.get("swap", False),
              "parity reference is not the unchanged causal_rollout of A@20k memory off at rho 1.0")
        _need(all(k_.startswith(f"{drv}|0|") for k_ in xr.get("noise_key", [""])) and xr.get("noise_key"), "parity reference is not seed 0")
        _need(xr.get("enc_sha") == c.get("enc_sha") and xr.get("ray_sha") == c.get("ray_sha"), "parity reference and cache saw different inputs")
        n = len(data["ticks"][drv])
        _need(len(xr["lat"]) == len(c["lat"]) == n, f"parity not whole-drive ({len(xr['lat'])}/{len(c['lat'])}/{n} ticks)")
        _need(all(torch.equal(xr[k_], c[k_]) for k_ in ("ticks", "src_ticks", "ok", "hard", "restart")) and xr.get("amp") == c.get("amp"), "parity: ticks / flags / amp differ")
        d = _maxdiff(xr["lat"], c["lat"])
        _need(math.isfinite(d) and d < 0.01, f"parity recomputed: max |diff| {d:.4g} >= 0.01")
        # ---- six PINNED dev references: full-file digests, signature, measured seed 0 on the complete five-tick prefix ----
        for sc in bind["dev"]:
            f = f"{bind['dev_ref']}/{sc}_s0.pt"
            _need(os.path.isfile(f), f"dev reference {f} missing")
            dbuf = read(f)
            _need(hb(dbuf) == P["dev_ref_sha256"][sc] == rec.get("dev_files", {}).get(sc), f"dev reference {f}: digest differs from the pinned known-good file")
            dr = tload(dbuf); del dbuf
            _need(str(dr.get("snapshot", "")).endswith(A_PATH) and dr.get("snapshot_step") == 20000 and dr.get("method") == "independent"
                  and dr.get("swap_from", -1) == -1 and dr.get("seed", 0) == 0, f"{f}: not the signed A@20k independent seed-0 render")
            sk = rec.get("dev_seed", {}).get(sc, {})
            _need(sk.get("file") and os.path.isfile(sk["file"]), f"{sc}: dev seed-check artifact missing")
            sbuf = read(sk["file"])
            _need(hb(sbuf) == sk.get("sha256"), f"{sc}: dev seed-check artifact differs from the record's digest")
            s_ = tload(sbuf); del sbuf
            _need(s_.get("ckpt_sha") == bind["ckpt_sha"] and s_.get("rho") == 1.0 and s_.get("method") == "causal_off" and s_.get("roll_code") == rollout_roll_code()
                  and s_.get("noise_key") and all(k_.startswith(f"{sc}|0|") for k_ in s_["noise_key"]), f"{sc}: seed-check artifact is not causal_rollout seed 0")
            k = len(s_["lat"]); nh = [i for i in range(k) if not bool(s_["hard"][i])]
            _need(k == 5 == len(dr["ticks"][:5]) and torch.equal(s_["ticks"], dr["ticks"][:5]) and torch.equal(s_["hard"].bool(), dr["hard"][:5].bool()),
                  f"{sc}: seed check is not the complete five-tick prefix with matching hard flags ({k} ticks)")
            _need(nh and len(s_["noise_key"]) == len(nh), f"{sc}: seed check has no / inconsistent non-hard ticks")
            ds = _maxdiff(s_["lat"][nh], dr["lat"][nh])
            _need(math.isfinite(ds) and ds < 0.01, f"{sc}: dev reference is not seed 0 (first ticks differ by {ds:.4g})")
        return True, "", rsha, lat
    except Bad as e:
        return False, str(e), None, None


def resume_identity(saved, cur, mix_from, start):
    """saved / cur: {"gate": gate record sha256 or None, "approved": approved cache record sha256 or None}; saved also "pin".
    -> "" if the checkpoint may resume under the current gate, else the reason"""
    s8 = lambda x: (x or "none")[:12]
    if saved.get("gate") != cur.get("gate"):
        return f"checkpoint was made under gate record {s8(saved.get('gate'))}, not {s8(cur.get('gate'))}"
    if saved.get("approved") != cur.get("approved"):
        return f"approved cache record changed ({s8(saved.get('approved'))} -> {s8(cur.get('approved'))}): passed vs pending or a different record"
    if mix_from and start > mix_from and not saved.get("pin"):
        return f"checkpoint at update {start} > mix_from {mix_from} has no pinned cache record"
    if saved.get("approved") and saved.get("pin") and saved["pin"] != saved["approved"]:
        return "pinned cache record is not the approved one"
    return ""
