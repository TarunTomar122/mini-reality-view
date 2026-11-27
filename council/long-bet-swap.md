# Long bet swap (one line each). Read council/long-bet-astra.md and council/long-bet-fable.md.
AGREED: C (one history frame) forked from C 4k with optimiser state, to 20k; A to 20k on identical row / noise draws;
own-history rows from WHOLE-DRIVE own rollouts (deployment sampler, s-level history, resets at hard breaks); 6 own + 6
true + 4 singles; 10% no-history; s ~ U[0, 0.7]; LRs unchanged; 5k checks on 6 dev drives (s = 0.3) vs equal-step A with
the headline = the smaller of the ID / W % reductions (4k: 9.4%) + car F1, recall, late drift, lingering ghosts + one
side-by-side video; one 30-drive exam + one council review at 20k.
OWNER RULING (applies): lingering ghosts never stop the run (Fable's "+9 twice" stop is dropped).
STARTED (superset of both): own rollouts of C@4k on 48 seeded train drives, 12 each at s = 0 / 0.1 / 0.3 / 0.5
(data/own_4k, 8 jobs on 2080 Tis).
OPEN - reply with your final choice, one line each:
1. Pool: all 48 drives at the four s values (Fable) or 6 rotating drives at s = 0.3 (Astra)? (48 is rendering anyway.)
2. Refresh: 8k / 12k / 16k (Fable; ~8 GPU-h each, training continues on the previous cache meanwhile) or 10k / 15k
   (Astra; training waits for the new cache)?
3. Early stop: Astra's (two consecutive checks with zero / negative headline gain, or two consecutive car-F1 deficits
   > 0.02) or Fable's (car F1 < -0.03 or late drift > +0.03 two checks running; smoothness at 10k and 15k no better
   than 4k -> next is a second history frame)? If split I will apply BOTH (stop on whichever fires).
Write to council/long-bet-swap-<you>.md (<= 60 words).
