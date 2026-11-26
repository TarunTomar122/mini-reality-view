# Amendment 1 (pairing tolerance 45 -> 50 ms): Fable

**APPROVE.**

- 50 ms is where the data bends: usable ticks jump from 54.1% to 94.5%, then gain only 1.4 points up to 55 ms. It is also the value I proposed before any data, so it is not tuned to results.
- The example offsets flip by about 50 ms per camera, which suggests frames arrive 50 or 100 ms apart, so the nearest one is up to ~50 ms away by design. (My inference, not checked.)

**Not shown; report before training:** drives per list whose longest unbroken run is >= 190 ticks at 50 ms. One mid-drive reject breaks a 19 s clip, and 33,063 / 204 = 162 ticks per drive on average, so drive length may be the real limit, not the tolerance.

If fewer than 12 final drives qualify, bring a second amendment (55 ms, or a shorter clip) instead of changing it silently.
