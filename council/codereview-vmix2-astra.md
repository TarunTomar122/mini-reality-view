# Astra vmix review of `1de750b`

- **Hold update 1.** I replayed the trainer’s resume branch locally with mocked checkpoint and cache I/O. A step-2000 checkpoint from a passed gate accepted a pending gate and a different verified record. `train_clean.py:229–250,349` ignores the saved gate digest and pins only at first use. Preserve the original approval across resume, reject a missing pin after step 2000, and test both cases.
- **Hash the bytes consumed.** `cache_verify.py:71,95–99` hashes and then reopens the record and each tensor file. An atomic replacement between opens can associate the approved digest with different contents. Hash and parse the same bytes or open file.
- **Bind current training encodings.** `cache_check.py:70` checks encoding hashes once. At admission, `train_clean.py:160` supplies only ticks and rays. A replacement encoding with identical ticks can pair stale generated history with new cameras or targets. Compare each cache’s encoding digest with the bytes loaded by the trainer.
- **Verify the measured parity.** T7b’s synthetic fixture passes the production verifier with invented parity and no reference artifact (`slurm/clean_tests.sh:99–120`). Bind and recheck the whole-drive reference against the cached drive before first use, including flags, AMP and measured max difference <0.01.
- **Bind all six dev references.** `cache_check.py:108–117` fixes completeness and finite MSE, but a renamed seed-1 render satisfies the same snapshot, method and tick checks. Bind the six signed seed-0 file digests.
- **Resolved by inspection.** The builder is unchanged. The checker forwards all parity inputs. The trainer checks A’s file digest, retries at the mix boundary and enforces an existing pin. The pure-key jitter control and the 8k own-prefix probe have runnable entry points.
- **Result gate.** Python and shell syntax passed locally. Before update 1, verify vpriv’s 8k/12k decisions and stop, frozen base/init/inputs, t*=1, matched T2 at ρ=1, T1–T7b on the final hashes, and 100 H200-3g updates with step time and peak GB. Suite 3756339 and cache check 3756338 were builder-reported only. A5 permits pending parity through update 2000.

SIGNED: no
