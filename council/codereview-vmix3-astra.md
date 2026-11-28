# Astra vmix review of `424127f`

- **1. Resume passes.** All eight identity cases passed locally. The trainer saves the gate digest, approved record digest and pin, and checks them before the next update.
- **2. Same-byte verification is partial.** Cache records, cache tensors and gate stdin now use the bytes hashed. `train_clean.py:151,183` still hashes and reopens A. `vclean_gate.py:93–98` also accepts the A1 subprocess’s success for a decision it read separately. Load A from the hashed buffer and compare the subprocess’s verified decision with the one recorded.
- **3. Encoding binding passes.** Admission compares each cache header and record digest with the digest of the encoding bytes the trainer parsed (`train_clean.py:141–142`, `cache_verify.py:131–132`).
- **4. Parity provenance remains open.** The verifier measures the whole-drive difference, but the candidate supplies both reference and digest. In a local CPU check it accepted a relabelled copy of the cache as its own reference, as constructed in `clean_tests.sh:119–121`. Bind the expected reference digest independently from a matched rollout before cache use.
- **5. Dev identity remains open.** `cache_verify.py:72–76` pins whatever files occupy the six paths at gate creation. A changed tail survives the prefix check. The same CPU check also passed with one copied tick per ten-tick dev drive (`cache_verify.py:165–170`). Pin the six known-good full-file digests independently and require the complete five-tick prefix with matching hard flags.
- Cache check 3756346 and clean suite 3756347 were builder-reported. Their outcomes remain unverified here.

SIGNED: no
