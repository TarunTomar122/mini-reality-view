# Astra review of 8a22c71
(a) Accept job 3755087 for the authorized training run on one saxa h200_3g.71gb slice. The 600-update run measured 0.438 s/update and 29.0 GB peak.
(b) Accept the strict T4 isolation bound of <1e-4. The batch-1 versus batch-4 fp32 difference was 2.64e-5. The independent reset, history and window assertions retain their thresholds.
(c) Four bit-identical scene-0043 renders would verify those four inputs. They cannot establish which code produced the other drives or whether every private-noise key used the same tick ID.
A full remake is unnecessary if job and input records bind each scored render to frozen A@20k, the intended encoding, rays and settings, and a known revision with equivalent off-mode math. Verify ticks[i] == i across each affected drive.
Re-render any unverified or mismatching files with frozen reviewed code and re-score each affected rho panel with the signed scorer. If provenance is missing panel-wide, remake that panel.
Require 12 full off-renders per decision-relevant rho (six dev drives × two seeds), bound to the signed score and A@20k comparator. Require eligible ID and W, positive comparator means and finite ratios outside 1.20 ± 0.0005.
SIGN-IF T1–T7 all pass (the existing 2080 Ti T2 pass counts), lineage is closed as above, and the trainer hash checks pass for rho 0.5 with both ratios below threshold or rho 0.8 after an admissible 0.5 miss and both 0.8 ratios below threshold. Start fresh from frozen A@20k on the authorized slice.
