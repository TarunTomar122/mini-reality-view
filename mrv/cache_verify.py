"""ONE verifier of an A5 cache_check record, shared by scripts/vclean_gate.py (passed state), scripts/train_clean.py (passed
and pending states, before the first cache read and on every resume) and scripts/cache_check.py (self-check of its output)
(council/codereview-vmix-astra.md items 1-3).

A "bind" fixes what a record must prove, independently of the record: the trainer's ACTUAL A@20k file digest, the cache dir,
the input paths (lists / enc / rays), the exact train and dev drive lists, and the reviewed sha256 of every file of the cache
chain (builder, checker, matched rollout + its dependencies, this verifier). verify() accepts a record only if
  * it is a passing cache_check record (no failures) of that cache dir, checkpoint and inputs, written by the reviewed checker,
    and the reviewed files are the ones on disk now;
  * its cache was rendered by the reviewed builder (roll_code) at rho 1.0, seed 0, unmerged, and covers EXACTLY the train drives;
  * whole-drive parity against the unchanged causal_rollout (memory off, rho 1.0, A@20k, its roll_code) passed: matched ticks /
    flags / amp, all ticks, max |diff| < 0.01;
  * every train and dev cache-to-key MSE is present and finite;
  * every cache file's current sha256 equals the record's; with data= (encodings' ticks / rays) every file is also loaded and
    checked for shape (n, 4, 32, 64), fp16, ticks aligned with the encoding, its ray digest and its signed fields
    (encoding digests are checked by cache_check; the trainer re-checks tick alignment).
"""
import hashlib, json, math, os

REVIEWED = ["scripts/build_cache.py", "scripts/cache_check.py", "scripts/causal_rollout.py", "mrv/causal_video.py", "mrv/sd.py", "mrv/cache_verify.py"]
A_PATH = "runs/longA/ckpt_20000.pt"


def sha(f):
    h = hashlib.sha256()
    with open(f, "rb") as fh:
        for b in iter(lambda: fh.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def file_code(files=REVIEWED):
    return {f: sha(f) for f in files}


def builder_roll_code():  # == scripts/build_cache.py ROLL_CODE
    return hashlib.sha256(b"".join(open(f, "rb").read() for f in ["scripts/build_cache.py", "mrv/sd.py"])).hexdigest()[:12]


def rollout_roll_code():  # == scripts/causal_rollout.py ROLL_CODE
    return hashlib.sha256(b"".join(open(f, "rb").read() for f in ["scripts/causal_rollout.py", "mrv/causal_video.py", "mrv/sd.py"])).hexdigest()[:12]


def ray_sha(t):  # == build_cache / causal_rollout RAY_SHA
    return hashlib.sha256(t.float().numpy().tobytes()).hexdigest()[:16]


def make_bind(ckpt, cache, lists, enc, rays, code):
    L = json.load(open(lists))
    return {"ckpt_sha": sha(ckpt)[:16], "cache": cache, "lists": lists, "enc": enc, "rays": rays, "train": sorted(L["train"]), "dev": list(L["dev"]),
            "code_sha256": {f: code[f] for f in REVIEWED}}


fin = lambda x: isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


class Bad(Exception):
    pass


def _need(c, m):
    if not c:
        raise Bad(m)


def verify(path, bind, data=None):
    """-> (ok, reason, record sha256, {scene: fp16 latents} if data else None). data = {"ticks": {scene: tensor}, "rays": {scene: tensor}}"""
    try:
        _need(len(bind["train"]) == 144, f"bind has {len(bind['train'])} train drives, A5 fixes 144")
        _need(path and os.path.isfile(path), f"no cache record at {path or '(none given)'}")
        rsha = sha(path); rec = json.load(open(path))
        _need(rec.get("kind") == "cache_check" and rec.get("pass") is True and rec.get("failures") == [], f"not a passing cache_check record (failures {rec.get('failures')})")
        for k in ("cache", "ckpt_sha", "lists", "enc", "rays"):
            _need(rec.get(k) == bind[k], f"record {k} {rec.get(k)!r} != bound {bind[k]!r}")
        cur = file_code()
        for f in REVIEWED:
            _need(cur[f] == bind["code_sha256"][f], f"{f} on disk is not the reviewed file")
            _need(rec.get("code_sha256", {}).get(f) == bind["code_sha256"][f], f"record was not produced with the reviewed {f}")
        _need(rec.get("roll_code") == builder_roll_code(), "cache not rendered by the reviewed builder (roll_code)")
        b = rec.get("build", {})
        _need(b.get("merge") is False and isinstance(b.get("batch"), int) and b.get("amp"), f"build settings {b}")
        _need(sorted(rec.get("files", {})) == bind["train"] and rec.get("n_files") == 144, "record does not cover exactly the 144 train drives")
        pa = rec.get("parity", {})
        _need(pa.get("drive") in bind["train"] and pa.get("exit") == 0 and pa.get("fields_equal") is True and pa.get("whole") is True
              and fin(pa.get("max_diff")) and pa["max_diff"] < 0.01, f"parity not passed on a whole train drive: {pa}")
        _need(pa.get("ref_ckpt_sha") == bind["ckpt_sha"] and pa.get("ref_rho") == 1.0 and pa.get("ref_method") == "causal_off"
              and pa.get("ref_roll_code") == rollout_roll_code(), "parity reference is not the unchanged causal_rollout of A@20k off at rho 1.0")
        m = rec.get("mse", {})
        _need(sorted(m.get("train_per_drive", {})) == bind["train"] and all(fin(v) for v in m["train_per_drive"].values()), "train MSE missing / non-finite")
        _need(sorted(m.get("dev_per_drive", {})) == sorted(bind["dev"]) and all(fin(v) for v in m["dev_per_drive"].values()), "dev MSE missing / non-finite")
        _need(fin(m.get("train_mean")) and fin(m.get("dev_mean")), "MSE means non-finite")
        lat = {} if data else None
        for sc in bind["train"]:
            f = f"{bind['cache']}/{sc}_s0.pt"
            _need(os.path.isfile(f) and sha(f) == rec["files"][sc], f"cache file {f} missing or its digest differs from the record")
            if data:
                import torch
                c = torch.load(f, map_location="cpu"); x = c["lat"]; tk = data["ticks"][sc]
                _need(x.dtype == torch.float16 and tuple(x.shape) == (len(tk), 4, 32, 64) and torch.equal(c["ticks"], tk), f"{f}: shape / dtype / ticks")
                _need(c.get("ckpt_sha") == bind["ckpt_sha"] and c.get("roll_code") == rec["roll_code"] and c.get("method") == "cache_off"
                      and c.get("rho") == 1.0 and c.get("seed") == 0 and c.get("merge") is False, f"{f}: signed fields")
                _need(c.get("ray_sha") == ray_sha(data["rays"][sc]), f"{f}: rays changed since the cache was built")
                lat[sc] = x
        return True, "", rsha, lat
    except Bad as e:
        return False, str(e), None, None
