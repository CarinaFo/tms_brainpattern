# ============================================================
# PSD + FOOOF theta/alpha peak detection, then symptom-change correlations
# ============================================================
#
# Pipeline:
#   1. Compute Welch PSD per parcel for the selected session of each patient.
#   2. Fit FOOOF and extract the strongest theta (4-7 Hz) and alpha (8-13 Hz)
#      peak per parcel, plus the aperiodic (1/f) offset and exponent.
#   3. Look at two prefrontal target parcels first.
#   4. If most patients have no theta peak there, scan all parcels.
#   5. Correlate peak frequency deltas and 1/f parameters with symptom change
#      (delta HADS-D) at:
#         - the parcel of interest (left DLPFC),
#         - the parcel with the most detected theta peaks,
#         - the parcel with the most detected alpha peaks.
#
# ============================================================

import os
from glob import glob
from pathlib import Path

import mne
import numpy as np
import pandas as pd
from fooof import FOOOF

import matplotlib.pyplot as plt
from scipy import stats

# ============================================================
# Settings
# ============================================================

SYSTEM = "windows"             # "windows" or "linux"
SESSION_SUFFIX = "_1"          # "_1" for session 1, "_2" for session 2

if SYSTEM == "linux":
    base_dir = Path("/home/carinaf/LabData")
else:
    base_dir = Path("L:")

project_dir = base_dir / "Lab_LucaC/Carina/canonical_hmm_finalsample"

source_path = project_dir / "source_reco_giles_parcel"

# patients included in Forster et al., 2026
patient_list_path = (
    project_dir
    / "prepared_data_giles_1Hz_3Hzfiltereddata"
    / "patients_fitted_for_this_hmm.csv"
)

hmm_dir = project_dir / "hmm_fits_05Hzcanonical_1Hzfiltered"

output_dir = project_dir / "fooof_theta_alpha_results"
output_dir.mkdir(parents=True, exist_ok=True)

# https://osl-dynamics.readthedocs.io/en/latest/parcellations/giles38.html
# Two prefrontal parcels of interest.
target_parcels = {
    "dorsomedial_pfc_l_idx26": 26,
    "dlpfc_l_idx28": 28,
}

# define freqency bands
theta_band = (4, 7)
alpha_band = (8, 13)

freq_range = [2, 39]           # range FOOOF is fitted over

fooof_settings = {
    "peak_width_limits": [1, 8],
    "max_n_peaks": 6,
    "min_peak_height": 0.01,
    "aperiodic_mode": "knee",
    "verbose": False,
}

# --- Settings for the symptom section --------------------------------------
N_STATES = 6                   # which HMM solution the clinical csv comes from
EXCLUDE_REPEATER = False       # drop patients who repeated TMS (ID contains "R")

# Parcel of interest for the symptom analysis. Set to a key of target_parcels.
PRIMARY_PARCEL_KEY = "dlpfc_l_idx28"
PRIMARY_PARCEL_IDX = target_parcels[PRIMARY_PARCEL_KEY]

# Delta peak frequency
THETA_REF_HZ = 5.0
ALPHA_REF_HZ = 10.0

# ============================================================
# Peak extraction + FOOOF
# ============================================================

def get_peak_freq_power_in_band(peaks, band):
    """
    Return (center_freq, power) of the strongest FOOOF peak inside `band`,
    or (nan, nan) if there is no peak in that band.

    FOOOF peak_params_ columns:
        0 = center frequency
        1 = peak power (above the aperiodic fit)
        2 = bandwidth
    """
    if peaks is None or len(peaks) == 0:
        return np.nan, np.nan

    peaks = np.asarray(peaks)
    in_band = (peaks[:, 0] >= band[0]) & (peaks[:, 0] <= band[1])
    band_peaks = peaks[in_band]

    if len(band_peaks) == 0:
        return np.nan, np.nan

    best = band_peaks[np.argmax(band_peaks[:, 1])]   # highest-power peak
    return float(best[0]), float(best[1])


def fit_fooof_one_psd(freqs, psd, parcel_idx, parcel_name=None):
    """Fit FOOOF to one parcel PSD and return theta/alpha + aperiodic info."""
    fm = FOOOF(**fooof_settings)
    fm.fit(freqs, psd, freq_range=freq_range)

    peaks = fm.peak_params_

    theta_freq, theta_power = get_peak_freq_power_in_band(peaks, theta_band)
    alpha_freq, alpha_power = get_peak_freq_power_in_band(peaks, alpha_band)

    # Aperiodic (1/f) params.
    #   knee mode  -> [offset, knee, exponent]
    #   fixed mode -> [offset, exponent]
    ap = np.asarray(fm.aperiodic_params_, dtype=float)
    if ap.size == 3:
        offset, knee, exponent = ap[0], ap[1], ap[2]
    elif ap.size == 2:
        offset, exponent = ap[0], ap[1]
        knee = np.nan
    else:
        offset = knee = exponent = np.nan

    return {
        "parcel_idx": parcel_idx,
        "parcel_name": parcel_name if parcel_name is not None else f"parcel_idx_{parcel_idx}",

        "theta_peak_frequency": theta_freq,
        "theta_peak_power": theta_power,
        "theta_peak_detected": not np.isnan(theta_freq),

        "alpha_peak_frequency": alpha_freq,
        "alpha_peak_power": alpha_power,
        "alpha_peak_detected": not np.isnan(alpha_freq),

        "aperiodic_offset": float(offset),
        "aperiodic_knee": float(knee),
        "aperiodic_exponent": float(exponent),

        "fooof_r_squared": fm.r_squared_,
        "fooof_error": fm.error_,
        "n_detected_peaks": 0 if peaks is None else len(peaks),
    }


def add_patient_id_from_session(df):
    """Add a patient_id column from session names like 'patientID_2'."""
    df = df.copy()
    df["patient_id"] = (
        df["session"].astype(str).str.rsplit("_", n=1).str[0]
    )
    return df


# ============================================================
# Helper functions: clinical data + statistics
# ============================================================

def load_and_prep_data(n_states, exclude_repeater=False):
    """Load the HMM + clinical summary csv for a given number of states."""
    csv_path = hmm_dir / f"hmm_demo_hads2704_{n_states}.csv"
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(df["patient"])
    print(f"Patients in HMM csv: {len(unique_ids)}")

    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        print(f"{len(repeater_ids)} patients repeated the treatment")
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].astype(str).str.contains("R")]
        np.save(hmm_dir / "dropped_indices.npy", np.array(repeater_positions))

    print(f"Analyzing {df['patient'].nunique()} patients after filtering")

    df["state"] = df["state"] + 1   # states starting from 1

    # broadcast demographics to every row of a patient
    for col in ["age", "gender", "responder", "group"]:
        if col in df.columns:
            df[col] = df.groupby("patient")[col].transform("first")

    for col in ["patient", "session", "tms", "state", "responder", "group", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category", errors="ignore")

    return df


def summarize_correlation(frame, x_col, y_col, label):
    """Spearman + Pearson correlation between two columns (rows with both present)."""
    sub = frame[[x_col, y_col]].dropna()
    out = {
        "test": "correlation", "label": label, "n": len(sub),
        "spearman_r": np.nan, "spearman_p": np.nan,
        "pearson_r": np.nan, "pearson_p": np.nan,
    }
    if len(sub) >= 3:
        sr, sp = stats.spearmanr(sub[x_col], sub[y_col])
        pr, pp = stats.pearsonr(sub[x_col], sub[y_col])
        out.update(spearman_r=sr, spearman_p=sp, pearson_r=pr, pearson_p=pp)
    return out


def regline_on_ax(ax, x, y, line_color="#C44E52", point_color="#4C72B0"):
    """Scatter points + least-squares regression line with a 95% CI band."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    ax.scatter(x, y, alpha=0.8, edgecolor="white", color=point_color)

    n = len(x)
    if n >= 3 and np.ptp(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xs = np.linspace(x.min(), x.max(), 100)
        ys = intercept + slope * xs
        ax.plot(xs, ys, color=line_color, lw=2)

        dof = n - 2
        sxx = np.sum((x - x.mean()) ** 2)
        if dof > 0 and sxx > 0:
            resid = y - (intercept + slope * x)
            s_err = np.sqrt(np.sum(resid ** 2) / dof)          # residual std
            tval = stats.t.ppf(0.975, dof)
            se_fit = s_err * np.sqrt(1.0 / n + (xs - x.mean()) ** 2 / sxx)
            ax.fill_between(xs, ys - tval * se_fit, ys + tval * se_fit,
                            color=line_color, alpha=0.15)
    return n


def annotate_corr(ax, x, y, fallback_title):
    """Title an axis with Pearson r and Spearman rho if computable."""
    sub_x = np.asarray(x, dtype=float)
    sub_y = np.asarray(y, dtype=float)
    mask = np.isfinite(sub_x) & np.isfinite(sub_y)
    if mask.sum() >= 3:
        pr, pp = stats.pearsonr(sub_x[mask], sub_y[mask])
        sr, sp = stats.spearmanr(sub_x[mask], sub_y[mask])
        ax.set_title(f"{fallback_title}\nr={pr:.2f} p={pp:.3f} | rho={sr:.2f} p={sp:.3f}",
                     fontsize=10)
    else:
        ax.set_title(fallback_title)


# ============================================================
# Find patient sessions
# ============================================================

patient_sessions = {}

for sess_dir in sorted(glob(str(source_path / "*_*"))):
    sess_id = os.path.basename(sess_dir)
    patient_id = sess_id.split("_")[0]

    src_path = source_path / sess_id / "parc" / "lcmv-parc-raw.fif"
    if not src_path.exists():
        continue

    patient_sessions.setdefault(patient_id, []).append({
        "session": sess_id,
        "src_path": str(src_path),
    })


# ============================================================
# Keep fitted patients, then the requested session only
# ============================================================

patient_ids = pd.read_csv(patient_list_path)

final_ids = [
    p for p in patient_sessions
    if p in patient_ids["patient_id"].values
]

sess_selected = []
for p in final_ids:
    sess_selected.extend([
        s for s in patient_sessions[p]
        if s["session"] == f"{p}{SESSION_SUFFIX}"
    ])

print(f"Number of selected sessions: {len(sess_selected)}")


# ============================================================
# Compute PSDs for all parcels and all patients
# ============================================================

all_psds = {}
freqs = None
parcel_names = None

for session in sess_selected:
    session_id = session["session"]
    src_path = session["src_path"]
    print(f"Reading {session_id}")

    raw = mne.io.read_raw_fif(src_path, preload=True)

    spectrum = raw.compute_psd(
        method="welch",
        picks="misc",          # all parcel channels
        fmin=1,
        fmax=40,
        n_fft=1000,
        reject_by_annotation=True,
    )

    psds, this_freqs = spectrum.get_data(return_freqs=True, picks='misc')

    # parcel names taken from the SAME picked channels, so row i of psds
    # always matches parcel_names[i]
    if parcel_names is None:
        parcel_names = spectrum.ch_names

    if freqs is None:
        freqs = this_freqs
    elif not np.allclose(freqs, this_freqs):
        raise ValueError(f"Frequency mismatch for {session_id}")

    all_psds[session_id] = psds
    print(f"{session_id}: PSD shape = {psds.shape}")


# ============================================================
# Step 1: fit only the target parcels
# ============================================================

target_results = []

for session_id, psds in all_psds.items():
    for parcel_name, parcel_idx in target_parcels.items():
        result = fit_fooof_one_psd(
            freqs=freqs,
            psd=psds[parcel_idx, :],
            parcel_idx=parcel_idx,
            parcel_name=parcel_name,
        )
        result["session"] = session_id
        result["analysis_scope"] = "target_parcels"
        target_results.append(result)

target_results = pd.DataFrame(target_results)
target_results = add_patient_id_from_session(target_results)

target_csv = output_dir / "fooof_target_parcels_theta_alpha.csv"
target_results.to_csv(target_csv, index=False)
print(f"Saved target parcel results to: {target_csv}")


# ============================================================
# Step 2: do most patients have NO theta peak in either target parcel?
# ============================================================

target_theta_summary = (
    target_results
    .groupby("session")
    .agg(
        n_target_parcels=("parcel_idx", "count"),
        n_target_theta_peaks=("theta_peak_detected", "sum"),
    )
    .reset_index()
)
target_theta_summary["no_theta_in_target_parcels"] = (
    target_theta_summary["n_target_theta_peaks"] == 0
)

n_patients = target_theta_summary["session"].nunique()
n_no_theta = target_theta_summary["no_theta_in_target_parcels"].sum()
prop_no_theta = n_no_theta / n_patients
majority_no_theta = prop_no_theta > 0.5

print(f"Patients with no theta peak in target parcels: {n_no_theta}/{n_patients}")
print(f"Proportion with no theta peak in target parcels: {prop_no_theta:.3f}")

summary_csv = output_dir / "target_parcel_theta_missing_summary.csv"
target_theta_summary.to_csv(summary_csv, index=False)
print(f"Saved target theta summary to: {summary_csv}")


# ============================================================
# Step 3: if most patients have no theta peak, scan all parcels
# ============================================================

all_parcel_results = None   # stays None unless we run the full scan


def compute_all_parcel_results():
    """Fit FOOOF for every parcel of every selected session."""
    rows = []
    for session_id, psds in all_psds.items():
        print(f"Fitting all parcels for {session_id}")
        for parcel_idx in range(psds.shape[0]):
            parcel_name = (
                parcel_names[parcel_idx]
                if parcel_names is not None and parcel_idx < len(parcel_names)
                else f"parcel_idx_{parcel_idx}"
            )
            res = fit_fooof_one_psd(freqs, psds[parcel_idx, :], parcel_idx, parcel_name)
            res["session"] = session_id
            res["analysis_scope"] = "all_parcels"
            rows.append(res)
    return add_patient_id_from_session(pd.DataFrame(rows))


if majority_no_theta:
    print("Majority have no theta peak in target parcels -> fitting all parcels...")
    all_parcel_results = compute_all_parcel_results()
    all_parcel_results.to_csv(output_dir / "fooof_all_parcels_theta_alpha.csv", index=False)

    # ---- Which parcels carry peaks in the most patients ----
    theta_detected = all_parcel_results[all_parcel_results["theta_peak_detected"]]
    alpha_detected = all_parcel_results[all_parcel_results["alpha_peak_detected"]]

    theta_peak_counts_by_parcel = (
        theta_detected
        .groupby(["parcel_idx", "parcel_name"])
        .agg(
            n_patients_with_theta_peak=("session", "nunique"),
            mean_theta_peak_power=("theta_peak_power", "mean"),
            mean_theta_peak_frequency=("theta_peak_frequency", "mean"),
        )
        .reset_index()
        .sort_values(["n_patients_with_theta_peak", "mean_theta_peak_power"],
                     ascending=[False, False])
    )
    alpha_peak_counts_by_parcel = (
        alpha_detected
        .groupby(["parcel_idx", "parcel_name"])
        .agg(
            n_patients_with_alpha_peak=("session", "nunique"),
            mean_alpha_peak_power=("alpha_peak_power", "mean"),
            mean_alpha_peak_frequency=("alpha_peak_frequency", "mean"),
        )
        .reset_index()
        .sort_values(["n_patients_with_alpha_peak", "mean_alpha_peak_power"],
                     ascending=[False, False])
    )

    theta_peak_counts_by_parcel.to_csv(output_dir / "theta_peak_counts_by_parcel.csv", index=False)
    alpha_peak_counts_by_parcel.to_csv(output_dir / "alpha_peak_counts_by_parcel.csv", index=False)

    print("\nTop theta parcels:")
    print(theta_peak_counts_by_parcel.head())
    print("\nTop alpha parcels:")
    print(alpha_peak_counts_by_parcel.head())
else:
    print("Most patients DO have a theta peak in the target parcels. Skipping all-parcel scan.")


# ============================================================
# Step 4: clinical / symptom data (one row per patient)
# ============================================================

df = load_and_prep_data(n_states=N_STATES, exclude_repeater=EXCLUDE_REPEATER)

# we want session 1 - session 3 HADS, currently session 3 minus session 1
df["delta_hads_d_1_to_3"] = -1 * (df["delta_hads_d_1_to_3"])

# keep first row only (multiple rows because of states etc.)
symptom_first = (
    df
    .dropna(subset=["patient", "delta_hads_d_1_to_3"])
    .drop_duplicates(subset="patient", keep="first")
    [["patient", "delta_hads_d_1_to_3"]]
    .copy()
)
symptom_first["patient"] = symptom_first["patient"].astype(str)
print(f"\nPatients with symptom-change data: {len(symptom_first)}")


# ============================================================
# Helper: symptom-change correlations at one parcel
# ============================================================

def symptom_correlations_at_parcel(source_df, parcel_idx, parcel_name,
                                   bands, scope_label):
    """
    Build a per-patient table for `parcel_idx` from `source_df`, merge with
    symptom change, and run symptom-change correlations for:
        - each band's peak-frequency delta (signed, from canonical centre)
        - the aperiodic 1/f offset
        - the aperiodic 1/f exponent
    Returns (merged_dataframe, results_dataframe).
    """
    if parcel_idx is None:
        print(f"[{scope_label}] no parcel available; skipping.")
        return None, pd.DataFrame()

    cols = ["patient_id", "session", "aperiodic_offset", "aperiodic_exponent"]
    for b in bands:
        cols += [f"{b}_peak_frequency", f"{b}_peak_power", f"{b}_peak_detected"]

    tab = source_df[source_df["parcel_idx"] == parcel_idx].copy()[cols]
    tab["patient_id"] = tab["patient_id"].astype(str)

    for b in bands:
        tab[f"{b}_freq_delta"] = tab[f"{b}_peak_frequency"]

    merged = symptom_first.merge(
        tab, left_on="patient", right_on="patient_id", how="inner"
    )

    tag = f"{scope_label}: parcel idx{parcel_idx} ({parcel_name})"
    n_peak = {b: int(merged[f"{b}_peak_detected"].sum()) for b in bands}
    print(f"\n[{tag}] merged rows = {len(merged)}; peaks detected here = {n_peak}")

    res = []
    for b in bands:
        res.append(summarize_correlation(
            merged, f"{b}_freq_delta", "delta_hads_d_1_to_3",
            f"{b} freq delta vs symptom change [{tag}]"))
    res.append(summarize_correlation(
        merged, "aperiodic_offset", "delta_hads_d_1_to_3",
        f"1/f offset vs symptom change [{tag}]"))
    res.append(summarize_correlation(
        merged, "aperiodic_exponent", "delta_hads_d_1_to_3",
        f"1/f exponent vs symptom change [{tag}]"))

    return merged, pd.DataFrame(res)


# ============================================================
# Step 5: symptom correlations at the PARCEL OF INTEREST (left DLPFC)
# ============================================================

stats_df, results_interest = symptom_correlations_at_parcel(
    source_df=target_results,
    parcel_idx=PRIMARY_PARCEL_IDX,
    parcel_name=PRIMARY_PARCEL_KEY,
    bands=["theta", "alpha"],
    scope_label="parcel_of_interest",
)
if stats_df is not None:
    stats_df.to_csv(output_dir / "symptom_peak_merged_interest.csv", index=False)


# ============================================================
# Step 6: symptom correlations at the PARCEL WITH MOST DETECTED PEAKS
#         (separately for theta and alpha)
# ============================================================

# need all-parcel fits for this; compute them if the Step 3 scan didn't run
if all_parcel_results is None:
    print("\nAll-parcel FOOOF results not present yet -> computing them now...")
    all_parcel_results = compute_all_parcel_results()
    all_parcel_results.to_csv(output_dir / "fooof_all_parcels_theta_alpha.csv", index=False)


def best_parcel_for_band(detected_col):
    """Parcel where the most unique patients have a detected peak."""
    det = all_parcel_results[all_parcel_results[detected_col]]
    if det.empty:
        return None, None, 0
    counts = (
        det.groupby(["parcel_idx", "parcel_name"])["session"]
        .nunique()
        .reset_index(name="n_patients_with_peak")
        .sort_values("n_patients_with_peak", ascending=False)
    )
    top = counts.iloc[0]
    return int(top["parcel_idx"]), top["parcel_name"], int(top["n_patients_with_peak"])


theta_parcel_idx, theta_parcel_name, theta_n = best_parcel_for_band("theta_peak_detected")
alpha_parcel_idx, alpha_parcel_name, alpha_n = best_parcel_for_band("alpha_peak_detected")
print(f"\nMost theta peaks: parcel {theta_parcel_idx} ({theta_parcel_name}) in {theta_n} patients")
print(f"Most alpha peaks: parcel {alpha_parcel_idx} ({alpha_parcel_name}) in {alpha_n} patients")

_, results_best_theta = symptom_correlations_at_parcel(
    source_df=all_parcel_results,
    parcel_idx=theta_parcel_idx,
    parcel_name=theta_parcel_name,
    bands=["theta"],
    scope_label="most_theta_peaks",
)
_, results_best_alpha = symptom_correlations_at_parcel(
    source_df=all_parcel_results,
    parcel_idx=alpha_parcel_idx,
    parcel_name=alpha_parcel_name,
    bands=["alpha"],
    scope_label="most_alpha_peaks",
)


# ============================================================
# Step 7: collect + save all correlation results
# ============================================================

all_corr_results = pd.concat(
    [results_interest, results_best_theta, results_best_alpha],
    ignore_index=True,
)
all_corr_results.to_csv(output_dir / "symptom_correlation_results.csv", index=False)

print("\n=== Symptom-change correlation results ===")
for _, r in all_corr_results.iterrows():
    print(r.to_dict())


# ============================================================
# Step 8: plots (parcel of interest)
# ============================================================

if stats_df is not None:
   # (a) histograms of the peak frequencies
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, ref, color in [
        (axes[0], "theta_freq_delta", THETA_REF_HZ, "#4C72B0"),
        (axes[1], "alpha_freq_delta", ALPHA_REF_HZ, "#C44E52"),
    ]:
        ax.hist(stats_df[col].dropna(), bins=12, color=color, edgecolor="white")
        ax.axvline(ref, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel("Hz")
    axes[0].set_ylabel("n patients")
    fig.tight_layout()
    fig.savefig(output_dir / "hist_frequency_deltas.png", dpi=150)
    plt.close(fig)

    # (b) regression: peak frequency vs symptom change
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, col, ref, title in [
        (axes[0], "theta_freq_delta", THETA_REF_HZ, "Theta"),
        (axes[1], "alpha_freq_delta", ALPHA_REF_HZ, "Alpha"),
    ]:
        sub = stats_df.dropna(subset=[col, "delta_hads_d_1_to_3"])
        regline_on_ax(ax, sub[col], sub["delta_hads_d_1_to_3"])
        ax.axvline(ref, color="k", linestyle="--", linewidth=1)
        ax.set_xlabel(f"{title} peak frequency (Hz)")
        ax.set_ylabel("delta HADS-D (1 -> 3)")
        annotate_corr(ax, sub[col], sub["delta_hads_d_1_to_3"],
                      f"{title} freq vs symptom change")
    fig.tight_layout()
    fig.savefig(output_dir / "reg_freqdelta_vs_symptom.png", dpi=300)
    plt.close(fig)

    # (c) regression: 1/f parameters vs symptom change
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8))
    for ax, col, title in [
        (axes[0], "aperiodic_offset", "1/f offset"),
        (axes[1], "aperiodic_exponent", "1/f exponent"),
    ]:
        sub = stats_df.dropna(subset=[col, "delta_hads_d_1_to_3"])
        regline_on_ax(ax, sub[col], sub["delta_hads_d_1_to_3"])
        ax.set_xlabel(title)
        ax.set_ylabel("delta HADS-D (1 -> 3)")
        annotate_corr(ax, sub[col], sub["delta_hads_d_1_to_3"],
                      f"{title} vs symptom change")
    fig.tight_layout()
    fig.savefig(output_dir / "reg_aperiodic_vs_symptom.png", dpi=300)
    plt.close(fig)

print(f"\nSaved plots and results to: {output_dir}")


# ============================================================
# Save example FOOOF fits for a few random patients
# ============================================================

EXAMPLE_PARCEL_IDX = 25   # which parcel to illustrate (e.g. 25 for the dominant-alpha parcel)
N_EXAMPLES = 5
RANDOM_SEED = 0                            # fixed seed -> same patients each run

example_dir = output_dir / "fooof_example_fits"
example_dir.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(RANDOM_SEED)
session_ids = list(all_psds.keys())
n_pick = min(N_EXAMPLES, len(session_ids))
example_sessions = rng.choice(session_ids, size=n_pick, replace=False)

for session_id in example_sessions:
    psds = all_psds[session_id]

    fm = FOOOF(**fooof_settings)
    fm.fit(freqs, psds[EXAMPLE_PARCEL_IDX, :], freq_range=freq_range)

    fig, ax = plt.subplots(figsize=(8, 5))
    # raw PSD + full fit + aperiodic fit, with detected peak centres marked
    fm.plot(ax=ax, plot_peaks="dot")
    n_peaks = 0 if fm.peak_params_ is None else len(fm.peak_params_)
    ax.set_title(f"{session_id} | parcel idx{EXAMPLE_PARCEL_IDX} | "
                 f"R2={fm.r_squared_:.2f} | {n_peaks} peak(s)")

    out_png = example_dir / f"fooof_fit_{session_id}_parcel{EXAMPLE_PARCEL_IDX}.png"
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"Saved example FOOOF fit: {out_png}")