"""Interactive drift annotation tool.

Loads all recordings, computes the PEL-PSG temporal offset series for each,
displays a scatter plot with fitted regression line, and lets you accept (y)
or reject (n) the drift estimate.  Accepted drift parameters are saved to a
JSON file that can later be loaded in detection_results.ipynb to correct
PEL interval timing.

Usage:
    python script/annotate_drift.py [--output drift_annotations.json]
                                    [--input drift_annotations.json]

Keys while viewing a plot:
    y / Enter  — accept this drift estimate
    n          — reject (no reliable drift detected)
    b          — go back to the previous recording
    q          — quit and save progress so far

Re-running with --input resumes from where you left off (already-annotated
recordings are skipped).
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import linregress
from sklearn.linear_model import RANSACRegressor
from tqdm import tqdm

# ── project imports ──────────────────────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

tqdm.pandas()

from cinc.notebook.detection_results import (
    get_structured_participant_data_df,
    apply_matching_processing,
)

# ── label maps (same as notebooks) ──────────────────────────────────────────
MATTRESS_LABELS_MAP = {
    "h1": "H1",
    "h2": "H2",
    "h3": "H3",
    "h3_sousmatelas": "H3UM",
    "h3_surcouche": "H3OL",
    "h4": "H4",
}

ID_LABELS_MAP = {
    "v1": "V1",
    "v2": "V2",
    "v3": "V3",
    "v4": "V4",
    "v5": "V5",
    "shanaz": "V6",
    "aurelie": "V7",
    "bastien": "V8",
}


# ── drift computation ───────────────────────────────────────────────────────
def compute_drift_series(matches):
    """Return (ref_times, offsets_ms) from matched PSG-PEL heartbeat pairs."""
    if not matches:
        return np.array([]), np.array([])

    ref_times, offsets_ms = [], []
    for m in matches:
        psg_mid = (m["rate1"].start_s + m["rate1"].end_s) / 2
        pel_mid = (m["rate2"].start_s + m["rate2"].end_s) / 2
        ref_times.append(psg_mid)
        offsets_ms.append((pel_mid - psg_mid) * 1000.0)

    ref_times = np.array(ref_times)
    offsets_ms = np.array(offsets_ms)
    order = np.argsort(ref_times)
    return ref_times[order], offsets_ms[order]


def unwrap_offsets(offsets_ms, heartbeat_period_ms):
    """Phase-unwrap the offset series to remove heartbeat-period jumps.

    When drift accumulates enough that the Hungarian matcher pairs a PSG beat
    with a PEL beat one cycle off, the offset jumps by approximately
    ±heartbeat_period_ms.  This function detects those jumps and removes them,
    producing a continuous offset series suitable for linear regression.

    The approach: walk through consecutive offsets.  When the difference between
    two consecutive offsets exceeds half a heartbeat period, we interpret it as
    a wrap event and apply a cumulative correction of the nearest integer
    multiple of heartbeat_period_ms.
    """
    if len(offsets_ms) < 2:
        return offsets_ms.copy()

    unwrapped = offsets_ms.copy()
    half_period = heartbeat_period_ms / 2.0
    cumulative_correction = 0.0

    for i in range(1, len(unwrapped)):
        diff = unwrapped[i] - unwrapped[i - 1]
        if abs(diff) > half_period:
            # Number of full heartbeat periods to correct
            n_jumps = round(diff / heartbeat_period_ms)
            cumulative_correction -= n_jumps * heartbeat_period_ms
        unwrapped[i] += cumulative_correction

    return unwrapped


def estimate_heartbeat_period_ms(matches):
    """Estimate the median heartbeat period from PSG intervals."""
    if not matches:
        return 800.0  # fallback ~75 bpm
    durations = [m["rate1"].duration_s for m in matches]
    return float(np.median(durations) * 1000.0)


def fit_drift(ref_times, offsets_ms, heartbeat_period_ms, residual_threshold_ms=50.0):
    """Fit drift with phase-unwrapping followed by RANSAC.

    1. Unwrap the offset series to undo heartbeat-period jumps caused by
       the matcher pairing wrong beats when drift is large.
    2. RANSAC on the unwrapped series for robustness against remaining outliers.
    3. linregress on inliers for r² and p-value.

    Returns:
        slope, intercept, r², p_value, inlier_mask, unwrapped_offsets
        (all None if too few points)
    """
    nones = (None, None, None, None, None, None)

    if len(ref_times) < 10:
        return nones

    # Phase-unwrap
    unwrapped = unwrap_offsets(offsets_ms, heartbeat_period_ms)

    # RANSAC on the unwrapped series
    X = ref_times.reshape(-1, 1)
    ransac = RANSACRegressor(
        residual_threshold=residual_threshold_ms,
        min_samples=0.5,
        max_trials=200,
    )
    try:
        ransac.fit(X, unwrapped)
    except ValueError:
        return nones

    inlier_mask = ransac.inlier_mask_
    if inlier_mask.sum() < 10:
        return nones

    # Precise stats on inliers
    result = linregress(ref_times[inlier_mask], unwrapped[inlier_mask])

    return (
        result.slope,
        result.intercept,
        result.rvalue ** 2,
        result.pvalue,
        inlier_mask,
        unwrapped,
    )


# ── data loading ─────────────────────────────────────────────────────────────
def load_data():
    print("Loading participant data...")
    df = get_structured_participant_data_df()

    df["mattress"] = df["mattress"].map(MATTRESS_LABELS_MAP)
    df = df.dropna(subset=["mattress"])
    df["id"] = df["id"].map(ID_LABELS_MAP)
    df = df.dropna(subset=["id"])

    print("Computing interval matching...")
    df = apply_matching_processing(df)
    return df


# ── recording key ────────────────────────────────────────────────────────────
def recording_key(row):
    return f"{row['id']}_{row['mattress']}_{row['position']}"


# ── interactive annotation ───────────────────────────────────────────────────
class DriftAnnotator:
    def __init__(self, df, existing_annotations):
        self.df = df
        self.annotations = dict(existing_annotations)
        self.decision = None

    def _on_key(self, event):
        if event.key in ("y", "enter"):
            self.decision = "accept"
            plt.close()
        elif event.key == "n":
            self.decision = "reject"
            plt.close()
        elif event.key == "b":
            self.decision = "back"
            plt.close()
        elif event.key == "q":
            self.decision = "quit"
            plt.close()

    def run(self):
        rows = list(self.df.iterrows())
        i = 0

        # Skip already-annotated recordings
        while i < len(rows) and recording_key(rows[i][1]) in self.annotations:
            i += 1

        if i > 0:
            print(f"Skipping {i} already-annotated recordings.")

        while i < len(rows):
            _, row = rows[i]
            key = recording_key(row)

            if key in self.annotations:
                i += 1
                continue

            matches = row.get("pel_cardiac_ensemble_interval_matches", [])
            ref_times, offsets_ms = compute_drift_series(matches)
            hb_period_ms = estimate_heartbeat_period_ms(matches)
            slope, intercept, r2, pval, inlier_mask, unwrapped = fit_drift(
                ref_times, offsets_ms, hb_period_ms
            )

            # ── draw plot (2 panels: raw offsets + unwrapped) ────────────
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            fig.canvas.mpl_connect("key_press_event", self._on_key)

            n_matches = len(ref_times)
            title = (
                f"{row['id']} / {row['mattress']} / {row['position']}  "
                f"({i + 1}/{len(rows)})  —  "
                f"HB period={hb_period_ms:.0f} ms"
            )

            # Left panel: raw offsets (shows wrapping)
            ax = axes[0]
            if n_matches > 0:
                ax.scatter(ref_times, offsets_ms, s=8, alpha=0.5, color="steelblue")
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Offset PEL-PSG (ms)")
            ax.set_title("Raw offsets", fontsize=9)

            # Right panel: unwrapped offsets + regression
            ax = axes[1]
            if unwrapped is not None and inlier_mask is not None:
                ax.scatter(
                    ref_times[~inlier_mask], unwrapped[~inlier_mask],
                    s=8, alpha=0.3, color="salmon", label=f"outliers ({(~inlier_mask).sum()})",
                )
                ax.scatter(
                    ref_times[inlier_mask], unwrapped[inlier_mask],
                    s=8, alpha=0.5, color="steelblue", label=f"inliers ({inlier_mask.sum()})",
                )
                ax.legend(fontsize=8, loc="upper left")
            elif n_matches > 0:
                ax.scatter(ref_times, offsets_ms, s=8, alpha=0.5, color="steelblue")

            if slope is not None:
                t_line = np.array([ref_times.min(), ref_times.max()])
                ax.plot(t_line, slope * t_line + intercept, "r-", linewidth=2)
                drift_ms_per_min = slope * 60
                accumulated_ms = abs(slope) * 180
                n_inliers = inlier_mask.sum() if inlier_mask is not None else n_matches
                ax.set_title(
                    f"Unwrapped — drift={drift_ms_per_min:.2f} ms/min, "
                    f"accum={accumulated_ms:.1f} ms, "
                    f"R²={r2:.3f}, inliers={n_inliers}/{n_matches}",
                    fontsize=9,
                )
            else:
                ax.set_title(f"Unwrapped — n={n_matches} (too few for regression)", fontsize=9)
            ax.set_xlabel("Time (s)")
            ax.set_ylabel("Offset PEL-PSG (ms)")

            fig.suptitle(title, fontsize=10)

            # Instructions
            fig.text(
                0.5, 0.01,
                "[y/Enter] accept   [n] reject   [b] back   [q] quit & save",
                ha="center", fontsize=11, color="gray",
            )

            self.decision = None
            plt.tight_layout(rect=[0, 0.04, 1, 1])
            plt.show(block=True)

            # ── handle decision ──────────────────────────────────────────
            if self.decision == "quit":
                print("Quitting early — progress saved.")
                break
            elif self.decision == "back":
                # Remove previous annotation and go back
                if i > 0:
                    i -= 1
                    prev_key = recording_key(rows[i][1])
                    self.annotations.pop(prev_key, None)
                    print(f"  ← Going back to {prev_key}")
                continue
            elif self.decision == "accept" and slope is not None:
                n_inliers = int(inlier_mask.sum()) if inlier_mask is not None else n_matches
                self.annotations[key] = {
                    "accepted": True,
                    "drift_rate_ms_per_s": slope,
                    "drift_intercept_ms": intercept,
                    "drift_rate_ms_per_min": slope * 60,
                    "drift_rate_ppm": slope * 1000,
                    "r_squared": r2,
                    "p_value": pval,
                    "n_matched_points": n_matches,
                    "n_inliers": n_inliers,
                }
                print(f"  ✓ {key}: drift={slope * 60:.2f} ms/min")
            else:
                # reject, or accept with no valid fit
                self.annotations[key] = {"accepted": False}
                print(f"  ✗ {key}: rejected")

            i += 1

        return self.annotations


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Interactive drift annotation tool")
    parser.add_argument(
        "--output", "-o",
        default="drift_annotations.json",
        help="Output JSON file (default: drift_annotations.json)",
    )
    parser.add_argument(
        "--input", "-i",
        default=None,
        help="Resume from an existing annotations file",
    )
    args = parser.parse_args()

    output_path = Path(args.output)

    # Load existing annotations if resuming
    existing = {}
    input_path = Path(args.input) if args.input else output_path
    if input_path.exists():
        with open(input_path, encoding="utf-8") as f:
            existing = json.load(f)
        print(f"Loaded {len(existing)} existing annotations from {input_path}")

    # Use TkAgg for interactive key events
    matplotlib.use("TkAgg")

    df = load_data()

    annotator = DriftAnnotator(df, existing)
    annotations = annotator.run()

    # Save
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(annotations, f, indent=2)

    n_accepted = sum(1 for v in annotations.values() if v.get("accepted"))
    n_rejected = sum(1 for v in annotations.values() if not v.get("accepted"))
    print(f"\nSaved {len(annotations)} annotations to {output_path}")
    print(f"  Accepted: {n_accepted}")
    print(f"  Rejected: {n_rejected}")


if __name__ == "__main__":
    main()
