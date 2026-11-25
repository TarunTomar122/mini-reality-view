# Council round 3: settle the last three points (short)

Read `council/s2-astra.md` and `council/s2-fable.md`. You already agree on: replay verification first (per-camera FOV,
ego origin, overlay checks), ~1-2k sim pairs from official train scenes with the ego hidden, fresh paired runs A
(real only) and B (sim first, then real) from identical init with paired real batches, a sim style row, phone-augmented
inputs, geometry = car-placement F1 + kerb error on held-out val scenes outside the final 30 with validation checks
(shifted annotations must score worse; shuffled cameras must score worse), and a go/no-go rule before scaling.

Practical constraint you must respect: there is no human annotator. Anything "marked by hand" must be done by an
automatic procedure (nuScenes 3D boxes and map projected into the drone view are exact; detecting cars/kerbs in our
generated images needs a model, e.g. a Cityscapes segmenter or detector, run by a script).

Decide, jointly if you can:
1. Include arm C (real + twin-key / sim rows mixed) in this pilot, or only after A vs B?
2. Pilot length and budget: 1k sim + 3k real (~4 GPU-h) or longer (~8 GPU-h)? Pick one, with the reason.
3. The exact automatic scoring procedure for generated images (which model, which classes, matching rule, the
   validation checks it must pass), given no human marking.

Write ONE file `council/s3-<your seat>.md`, at most half a page: your final answer to 1-3 and, if you still disagree
with the other seat on any point, one line saying why. Do not edit other files, run jobs or touch the cluster.
Finish by stating the file path you wrote.
