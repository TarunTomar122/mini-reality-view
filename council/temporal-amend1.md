# Amendment request 1 to PLAN-temporal.md (pairing tolerance), with data

The signed plan rejects a tick if any of the six nearest camera sweeps is more than 45 ms from the tick time
(Astra's round-2 bound; Fable proposed 50 ms; the merge took 45). Measured on all 204 rendered drives (33,063 ticks,
data/temporal_manifest.json):

- max over 6 cameras of |sweep - tick| (ms), percentiles 50 / 75 / 90 / 99: 44.5 / 48.8 / 49.8 / 78.5
- per-camera |offset| percentiles 50 / 90 / 99: 21.3 / 44.0 / 50.3
- usable ticks: 45 ms 54.1%, 50 ms 94.5%, 55 ms 95.9%, 60 ms 96.4%, 70 ms 98.0%
- example offsets of six consecutive ticks (ms): [-43.8 -36.2 -28.4 -1.3 -11.2 -20.8], [-43.7 ...], [6.7 14.0 21.9 -1.0 -10.8 -20.5],
  [6.6 14.2 22.0 49.1 39.2 29.6], ...

The nearest sweep is at most ~half a camera period away, so 45 ms sits in the middle of the natural distribution:
46% of ticks are rejected, sequences break about every other tick, and the >= 19 s contiguous clips needed for the
drift metric almost never exist. Proposed change: tolerance 50 ms (Fable's round-1/2 value), everything else unchanged
(offsets are still logged and reported; sequences still break at rejects and repeated bundles). At 50 ms the remaining
error is up to ~0.5 m at 10 m/s, as Fable estimated.
