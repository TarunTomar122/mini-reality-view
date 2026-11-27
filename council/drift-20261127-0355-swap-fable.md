# Drift check 8 swap: Fable (final)

**(i) Accept (hold): t = floor((1 - u^3) x 962).** Checked: `sd_render.py:77` is the stock SD-1.5 DDIM, 25 steps = t 1..961; P(t > 961) under R's draw = 0.038^(1/3) = 34%. A third of a 1k-update budget trains steps sampling never runs. A fix, not a tuning choice; gate t700 / t900 untouched.

**(ii) Accept (hold): divide each row's squared error by alpha-bar_t, before both means.** F is the last arm; a fail under a recipe whose signal sits in low-noise rows closes the track on "not learned" again. Stands without (i), but weaker: t > 961 rows would then carry ~220x weight. Keep clip 1.0 as R; it will fire on most updates (AdamW is scale-free), so report clipped share, not gate on it.
