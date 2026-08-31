"""Flag high-training-loss patients from a completed run, to build the
"filtered/cleaned dataset" variant for Pipeline-A-style replication.

Reference paper (Henry et al., BraTS 2020, section 2.6): "The filtering process
was based on previous training runs: cases with high training loss at the end
of the training procedure were flagged as potentially wrong and removed from
the complete training set, thus creating a 'cleaned' version of the training
dataset." The paper does not state an exact threshold; this script uses a
top-N-percent cutoff on each patient's *last recorded* training-split loss
(default: top 10%), which is a reasonable, documented reading of "high
training loss at the end of training" -- adjust --top_pct, or hand-edit the
output file, as you see fit.

Usage (after a normal, unfiltered baseline run has finished -- e.g. the
"variant 1" CascadedSEUnet run for a given fold):

    python -m src.find_noisy_patients \\
        --perf_csv runs/<run_folder>/patients_indiv_perf.csv \\
        --top_pct 10 \\
        --out excluded_patients_fold0.txt

Then pass that file to the "variant 2" (filtered) run for the *same fold*:

    python -m src.train ... --fold 0 --exclude_patients excluded_patients_fold0.txt
"""
import argparse

import pandas as pd

parser = argparse.ArgumentParser(description="Flag high-training-loss patients for exclusion")
parser.add_argument('--perf_csv', required=True, type=str,
                    help="path to a patients_indiv_perf.csv written by src.train")
parser.add_argument('--top_pct', default=10.0, type=float,
                    help="percentage (0-100) of patients with the highest final training loss "
                         "to flag (default: 10.0)")
parser.add_argument('--out', required=True, type=str,
                    help="output path: one flagged patient-dir name per line")


def main(args):
    df = pd.read_csv(args.perf_csv)
    train_rows = df[df["split"] == "train"]
    if train_rows.empty:
        raise ValueError(f"No split=='train' rows found in {args.perf_csv} -- was this CSV "
                         f"produced by src.train with patients_perf enabled?")

    # "at the end of the training procedure": keep each patient's last recorded epoch only.
    last_epoch_per_patient = train_rows.sort_values("epoch").groupby("id").tail(1)

    n_flag = max(1, round(len(last_epoch_per_patient) * args.top_pct / 100.0))
    flagged = last_epoch_per_patient.sort_values("loss", ascending=False).head(n_flag)

    print(f"{len(last_epoch_per_patient)} patients seen; flagging the {n_flag} "
          f"({args.top_pct}%) with the highest final training loss:")
    print(flagged[["id", "epoch", "loss"]].to_string(index=False))

    with open(args.out, "w") as f:
        for patient_id in flagged["id"]:
            f.write(f"{patient_id}\n")
    print(f"Wrote {len(flagged)} patient ID(s) to {args.out}")


if __name__ == '__main__':
    main(parser.parse_args())
