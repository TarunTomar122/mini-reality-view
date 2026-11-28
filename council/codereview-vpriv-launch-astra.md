# Astra launch review of 7518872
Keep the launcher held. The four pulled panels support A3 and select two private clips.
At 8k/12k/16k the smaller reductions are −0.511/−0.480/−0.577. None passes. At 16k, F1 −0.0105 and late gap +0.0016 meet safety (six and five contributing drives respectively). Own-cold ID +0.0453 and W +2.8717 select two.
The local tool hashes match. The signed comparator enforces paired seeds, rebuilds the k=3 list and receives floor 0 and cold reference B.

1. `scripts/a3_trigger.py:19–34` compares filenames echoed by the comparator. V8k metrics copied to the V16k filename pass identity, seed and scorer checks. Verify the metrics’ roll and cold paths and exact dev drive IDs, and score the same immutable bytes whose digests are recorded. Use explicit checks that survive Python optimization.
2. `scripts/a3_trigger.py:52` overwrites the launch record on every successful dry run. A rerun after submission erases the job ID. `slurm/launch_vpriv.sh:41–42` also reports success if attaching the job ID fails. Give each launch an immutable panel record, keep dry runs separate, and check the job receipt write before reporting success.
3. `slurm/launch_vpriv.sh:16–17` serializes launchers, but a failed queue query looks empty. A retry before the first job writes args.json can submit twice. Abort on queue errors and check the recorded submission under the lock.
4. Lines 19–26 check the final result and listed hashes, but accept any matching job directory. Verify the suite/T2/A/B IDs (3755915/3755760/3755758/3755759) and complete suite/A/B manifests against the reviewed deployment, including the launcher and trigger separately.
Suite 3755915 was reported through T3b with T5–T7 pending. Final suite success, deployed hashes and timely A1 scoring/stop coverage remain unverified locally. If the batch script changes, submit the corrected launcher held in place of the old copy.

SIGNED: no
