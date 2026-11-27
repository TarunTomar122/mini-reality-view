Accept the seeded initialization for the A@20k and vpriv cold-baseline gates. Neither gate depends on reproducing vlive’s initialization.
Report any vpriv–vlive difference as a joint effect of policy and initialization. A shared output at update 0 does not imply shared gradients.
Require a vlive-policy run with the same init_sha and matched draws only if claiming a policy-specific effect.
SIGNED: accept
