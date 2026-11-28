At 98ec4f8, the watcher records 12.early before the inline call and waits for it to return. Its next pass cannot consume the trainer slot during the attempt.
An early failure is logged and the watcher resumes scheduling the regular 12k scorer. If sbatch returned no valid job ID, CONT_PENDING blocks a second submission, including after a definite rejection.
Manual reconciliation is required before the regular scorer reaches that guard to let it continue training. Watcher 3756105 is builder-reported.
OK
