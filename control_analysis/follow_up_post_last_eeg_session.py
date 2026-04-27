"""Follow-up predictions from session 3
Author: Carina Forster

"""
import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.formula.api as smf
import matplotlib.pyplot as plt

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42

# linux doesn't have Arial
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# run in base python (3.12)
system='windows'

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/LabData')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:")
else:
    "No available system path defined *windows* or *linux*"

# ------------ Directories -------------#
# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')
fig_dir = Path(f"{hmm_dir}/figures")


def load_and_prep_data(n_states, exclude_repeater: bool = False):
    """
    Load and preprocess HMM demo questionnaire data.
    Drops patients with missing baseline HADS-D (session 1, pre).
    """
    csv_path = Path(f"{hmm_dir}/hmm_demo_hads_{n_states}.csv")
    df = pd.read_csv(csv_path)

    if exclude_repeater and "patient" in df.columns:
        df = df[~df["patient"].astype(str).str.contains("R")]

    if "state" in df.columns:
        df["state"] = df["state"] + 1

    for col in ["age", "gender", "responder", "group", "years_with_depression"]:
        if col in df.columns:
            df[col] = df.groupby("patient")[col].transform("first")

    # Drop patients with no baseline HADS-D
    required = {"session", "tms", "hads_dep_total", "patient"}
    if required.issubset(df.columns):
        baseline = df[
            (df["session"].astype(str) == "1") &
            (df["tms"].astype(str) == "pre")
        ].copy()

        missing_baseline = baseline.loc[
            baseline["hads_dep_total"].isna(), "patient"
        ].astype(str).unique()

        if len(missing_baseline) > 0:
            print("\nDropping patients with no baseline HADS-D before treatment:")
            print(", ".join(sorted(missing_baseline)))
            df = df[~df["patient"].astype(str).isin(missing_baseline)].copy()

    print(f"Analyzing {df['patient'].nunique()} patients")

    for col in ["patient", "session", "tms", "state", "responder", "group", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category", errors="ignore")

    return df

def prepare_mid_to_end_prediction_data(
    n_states: int,
    state_for_reg=(1, 2),
    symptom_col: str = "hads_dep_total",
    scale_deltafo_by_100: bool = True,
):
    """
    ΔFO at session 2 → predict final (last available) HADS

    Controls for:
        - baseline at session 2
        - days from session 2 EEG to final HADS
    """
    df = load_and_prep_data(n_states)

    future_path = Path(f"{hmm_dir}/future_hads_after_final_eeg_{n_states}.csv")
    future_df = pd.read_csv(future_path)

    valid_patients = set(df["patient"].astype(str).unique())
    future_df = future_df[future_df["patient"].astype(str).isin(valid_patients)].copy()

    st1, st2 = state_for_reg
    dfo1 = f"delta_fo_state{st1}"
    dfo2 = f"delta_fo_state{st2}"

    # --- identify future HADS columns ---
    if "future_depression" in future_df.columns:
        future_symptom_col = "future_depression"
    else:
        raise ValueError("Expected 'future_depression' column.")

    if "future_hads_date" in future_df.columns:
        date_col = "future_hads_date"
    else:
        raise ValueError("Expected 'future_hads_date' column.")

    future_df[date_col] = pd.to_datetime(future_df[date_col], errors="coerce")
    future_df = future_df.dropna(subset=[date_col])

    # --- get LAST HADS per patient ---
    future_last = (
        future_df.sort_values(["patient", date_col])
        .groupby("patient", as_index=False)
        .tail(1)
        .copy()
    )

    # --- symptoms at session 2 ---
    sym_wide = (
        df.pivot_table(index="patient", columns="session", values=symptom_col)
        .reset_index()
        .rename(columns={1: "sym_s1", 2: "sym_s2", 3: "sym_s3",
                         "1": "sym_s1", "2": "sym_s2", "3": "sym_s3"})
    )

    cov = df[["patient", "age", "gender"]].drop_duplicates()

    # --- compute ΔFO ---
    fo_prepost = (
        df[df["tms"].isin(["pre", "post"])]
        .groupby(["patient", "session", "state", "tms"])["fo"]
        .mean()
        .reset_index()
    )

    fo_wide_tms = (
        fo_prepost
        .pivot_table(index=["patient", "session", "state"], columns="tms", values="fo")
        .reset_index()
    )

    fo_wide_tms["delta_fo"] = fo_wide_tms["pre"].astype(float) - fo_wide_tms["post"].astype(float)
    fo_wide_tms = fo_wide_tms[fo_wide_tms["state"].isin(state_for_reg)].copy()

    # --- select session 2 ΔFO ---
    fo_sess2 = fo_wide_tms[fo_wide_tms["session"].astype(int) == 3].copy()

    fo_sess2_wide = (
        fo_sess2
        .pivot(index="patient", columns="state", values="delta_fo")
        .reset_index()
        .rename(columns={st1: dfo1, st2: dfo2})
    )

    # --- get session 2 EEG time ---
    eeg_time = (
        df[df["session"].astype(int) == 3]
        .groupby("patient")["session_datetime"]
        .min()
        .reset_index()
        .rename(columns={"session_datetime": "eeg_s3_datetime"})
    )

    # --- merge everything ---
    mid_df = (
        future_last
        .merge(fo_sess2_wide, on="patient", how="inner")
        .merge(sym_wide[["patient", "sym_s2"]], on="patient", how="left")
        .merge(eeg_time, on="patient", how="left")
        .merge(cov, on="patient", how="left")
        .rename(columns={
            "sym_s2": "sym_base",
            future_symptom_col: "sym_last",
            date_col: "last_hads_date"
        })
    )

    # --- compute time difference ---
    mid_df["days_after_s3"] = (
        pd.to_datetime(mid_df["last_hads_date"]) -
        pd.to_datetime(mid_df["eeg_s3_datetime"])
    ).dt.total_seconds() / (60 * 60 * 24)

    # --- clean ---
    mid_df = mid_df.dropna(
        subset=[dfo1, dfo2, "sym_base", "sym_last", "days_after_s3", "age", "gender"]
    ).copy()

    if scale_deltafo_by_100:
        mid_df[dfo1] *= 100
        mid_df[dfo2] *= 100

    return mid_df

def fit_mid_to_end_model(mid_df, state_for_reg=(1, 2), robust="HC3"):
    st1, st2 = state_for_reg
    dfo1 = f"delta_fo_state{st1}"
    dfo2 = f"delta_fo_state{st2}"

    model = smf.ols(
        f"sym_last ~ {dfo1} + {dfo2} * days_after_s3  + sym_base + age + C(gender)",
        data=mid_df
    ).fit(cov_type=robust)

    return model

model.summary()

def plot_mid_to_end_prediction(mid_df, model, state_for_reg=(1, 2), savepath=None):
    st1, st2 = state_for_reg
    dfo1 = f"delta_fo_state{st1}"
    dfo2 = f"delta_fo_state{st2}"

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), constrained_layout=True)

    def plot_panel(ax, x_term, other_term, title):
        ax.scatter(mid_df[x_term], mid_df["sym_last"], alpha=0.5, color="black")

        x_vals = np.linspace(mid_df[x_term].min(), mid_df[x_term].max(), 200)

        pred_df = pd.DataFrame({
            x_term: x_vals,
            other_term: mid_df[other_term].mean(),
            "sym_base": mid_df["sym_base"].mean(),
            "days_after_s3": mid_df["days_after_s3"].mean(),
            "age": mid_df["age"].mean(),
            "gender": mid_df["gender"].mode()[0],
        })

        yhat = model.predict(pred_df)

        ax.plot(x_vals, yhat, color="black", linewidth=3)
        ax.set_title(title)
        ax.set_xlabel("% ΔFO at session 2")
        ax.set_ylabel("Final HADS-D")

    plot_panel(axes[0], dfo1, dfo2, f"State {st1}")
    plot_panel(axes[1], dfo2, dfo1, f"State {st2}")

    if savepath:
        fig.savefig(savepath + ".png", dpi=300)

    return fig