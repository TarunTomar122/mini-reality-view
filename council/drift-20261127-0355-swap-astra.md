(i) accept — Use the 962 multiplier and clamp at 961. R spent about a third of its draws above the highest DDIM-25 render timestep. Keep the same seeded draws.
(ii) keep-R — Keep R’s epsilon loss. The map-off errors at t700 and t900 measure prediction error. They do not establish which rows dominate the branch gradients, and dividing by alpha-bar could push more updates into gradient clipping.
