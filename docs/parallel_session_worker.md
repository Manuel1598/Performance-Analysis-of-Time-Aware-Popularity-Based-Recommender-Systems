# Parallel session validation on a second computer

The validation-first protocol can split Top-N and session tuning across two
computers. Only validation tuning is distributed. Final test measurements and
the sequence-length ablation remain on the main computer so that all reported
runtimes use the same hardware.

## Prepare the second computer

1. Check out the same Git commit as the main computer.
2. Create the Python environment from requirements.txt. The launcher uses
   .venv-vsknn\Scripts\python.exe when it exists and otherwise uses python.
3. Copy these processed dataset directories to the same relative location:
   - data\recbole\adressa_recbole_sample
   - data\recbole\globo_recbole_sample
   - data\recbole\yoochoose_recbole_sample
4. Do not copy the main computer's active validation_trials.csv into the worker
   output directory.

## Start the worker

From a PowerShell terminal in the project directory:

    .\tools\run_session_validation_worker.ps1 -WorkerName session_pc2 -Device cpu

The worker runs only the three session datasets and writes to:

    recbole_results\validation_first_workers\session_pc2\validation_trials.csv

The run is resumable. Starting the same command again skips successful rows in
that worker file. Use CPU unless a separate, documented tuning-device decision
has been made. Final tests are CPU-only on the main computer regardless.

## Keep the main computer on Top-N tuning

Once the session worker is active, stop the all-scenarios process on the main
computer and restart its resumable tuning with this Top-N-only command:

    .\.venv-vsknn\Scripts\python.exe tools\run_validation_first_experiments.py `
      --phase tune --scenario topn --device cpu

## Merge the results

First stop every process that can write either input or the destination file.
Copy the worker CSV to the main computer. Then run:

    .\.venv-vsknn\Scripts\python.exe tools\merge_validation_first_results.py `
      recbole_results\validation_first\validation_trials.csv `
      recbole_results\validation_first_workers\session_pc2\validation_trials.csv `
      --output recbole_results\validation_first\validation_trials.csv

The output replacement is atomic. Exact duplicate rows are retained once. If
the same run_id has different values in the two files, the merge stops and
reports the conflicting columns.
Rows from older protocol versions already present in the main file are kept
unchanged.

After merging, restart the main runner so that it reads the combined IDs. The
main computer must be limited to Top-N tuning while the session worker is
active; otherwise both machines can calculate the same session configurations.
