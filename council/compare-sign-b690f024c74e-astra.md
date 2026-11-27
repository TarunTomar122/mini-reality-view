NO
At `b690f024c74e`, replacing both `per_seed` maps in a generated count list with `{}` produced empty ID and W comparator counts under both default Python and `python -O`. Both metrics still reported `can_pass: true` at floor 1, although the counting drive had three finite matches per seed.
Recompute each drive’s per-seed eligible and matched counts from the comparator and reject lists that disagree.
