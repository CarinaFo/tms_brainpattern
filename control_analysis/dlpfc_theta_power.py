# Beta power from ACC and symptom improvement
import os
import pandas as pd
from glob import glob
import mne
import numpy as np
from pathlib import Path

from scipy.stats import zscore
import statsmodels.formula.api as smf

import matplotlib.pyplot as plt
import seaborn as sns

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

# windows paths
preproc_path = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/preprocessed")
source_path  = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/source_reco_giles_parcel")

# load source data
patient_sessions = {}

for sess_dir in sorted(glob(os.path.join(preproc_path, "*_*")))[:-1]:

    session = os.path.basename(sess_dir)

    patient = session.split("_")[0]

    # Source parcel data
    src_path = os.path.join(
        source_path, session, "parc", "lcmv-parc-raw.fif"
    )
    if not os.path.exists(src_path):
        continue

    patient_sessions.setdefault(patient, []).append({
        "session": session,
        "src_path": src_path
    })


# load patient ID list
patient_ids = pd.read_csv(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"))


# Choose parcels after checking Giles38 labels
# https://osl-dynamics.readthedocs.io/en/latest/parcellations/giles38.html
parcels = {
    "inf_frontal_l": 22,
    "inf_frontal_r": 23,   # update if needed
    "acc_l": 34,
    "acc_r": 35,      # left anterior ACC
    "acc_midline": 37 
}

bands = {
    "theta": (4, 8),
    "alpha": (8, 12),
    "beta":  (13, 30)
}

def get_power_per_parcel(parcels: dict, bands: dict):

    rows = []

    for patient in patient_ids["patient_id"]:

        for sess_idx, sess in enumerate(patient_sessions[patient], start=1):

            raw_src = mne.io.read_raw_fif(
                sess["src_path"],
                preload=True,
                verbose=False
            )

            for region, parcelnum in parcels.items():

                parcel_raw = raw_src.copy().pick_channels([f"parcel_{parcelnum}"])

                for band_name, (fmin, fmax) in bands.items():

                    psd = parcel_raw.compute_psd(
                        method="welch",
                        fmin=fmin,
                        fmax=fmax,
                        picks="misc",
                        reject_by_annotation=True
                    )

                    rows.append({
                        "patient": str(patient),
                        "session": sess_idx,
                        "region": region,
                        "band": band_name,
                        "power": np.mean(psd.get_data(picks="misc"))
                    })

    pow_df = pd.DataFrame(rows)
    pow_df["log_power"] = np.log10(pow_df["power"])

    return pow_df


def plot_power_over_sessions():

    session_pairs = [
    (1, 2, "early"),
    (3, 4, "mid"),
    (5, 6, "late")
    ]

    results = []

    for region in list(parcels.keys()):

        for band in ["theta", "alpha", "beta"]:

            for pre_sess, post_sess, stage in session_pairs:

                pre_vals = []
                post_vals = []

                for patient_id in pow_df["patient"].unique():

                    patient_df = pow_df[
                        (pow_df["patient"] == patient_id) &
                        (pow_df["region"] == region) &
                        (pow_df["band"] == band)
                    ]

                    pre = patient_df.loc[
                        patient_df["session"] == pre_sess,
                        "log_power"
                    ]

                    post = patient_df.loc[
                        patient_df["session"] == post_sess,
                        "log_power"
                    ]

                    if pre.empty or post.empty:
                        continue

                    pre_vals.append(pre.iloc[0])
                    post_vals.append(post.iloc[0])

                stat, p = wilcoxon(pre_vals, post_vals)

                print(
                    f"{region} | {band} | {stage}: "
                    f"p={p:.4f}"
                )

                results.append({
                    "region": region,
                    "band": band,
                    "stage": stage,
                    "p": p
                })

                # -------------------------
                # Plot
                # -------------------------
                plot_df = pd.DataFrame({
                    "PRE": pre_vals,
                    "POST": post_vals
                })

                plot_df = plot_df.melt(
                    var_name="condition",
                    value_name="log_power"
                )

                plt.figure(figsize=(4,5))

                sns.boxplot(
                    data=plot_df,
                    x="condition",
                    y="log_power"
                )

                sns.stripplot(
                    data=plot_df,
                    x="condition",
                    y="log_power",
                    color="black",
                    alpha=0.4
                )

                plt.title(
                    f"{region.upper()} | "
                    f"{band.upper()} | "
                    f"{stage}\n"
                    f"p={p:.4f}"
                )

                plt.tight_layout()
                plt.show()  


def predict_symptom_improvement(pow_df):

    
    df_clin = load_and_prep_data(10)

    theta_df = pow_df[(pow_df['band'] == 'theta') & (pow_df['region'] == 'inf_frontal_l')]

    session_pairs = [
        # EEG pre, EEG post, HADS pre, HADS post, stage
        (1, 2, 1, 2, "early"),
        (3, 4, 2, 3, "mid")
    ]

    rows = []

    for pre_eeg, post_eeg, hads_pre_sess, hads_post_sess, stage in session_pairs:

        for patient_id in theta_df["patient"].unique():

            # ---------------------------------
            # Theta change
            # ---------------------------------
            patient_theta = theta_df[
                theta_df["patient"] == patient_id
            ]

            pre_theta = patient_theta.loc[
                patient_theta["session"] == pre_eeg,
                "log_power"
            ]

            post_theta = patient_theta.loc[
                patient_theta["session"] == post_eeg,
                "log_power"
            ]

            # ---------------------------------
            # Clinical data
            # ---------------------------------
            patient_clin = df_clin[
                df_clin["patient"].astype(str) == str(patient_id)
            ]

            baseline_hads = patient_clin.loc[
                (patient_clin["session"] == hads_pre_sess) &
                (patient_clin["tms"] == "pre"),
                "hads_dep_total"
            ]

            followup_hads = patient_clin.loc[
                (patient_clin["session"] == hads_post_sess) &
                (patient_clin["tms"] == "pre"),
                "hads_dep_total"
            ]

            age = patient_clin["age"].iloc[0]
            gender = patient_clin["gender"].iloc[0]

            if (
                pre_theta.empty or
                post_theta.empty or
                baseline_hads.empty or
                followup_hads.empty
            ):
                continue

            rows.append({

                "patient_id": patient_id,
                "stage": stage,

                # predictor
                "delta_theta":
                    post_theta.iloc[0] - pre_theta.iloc[0],

                # outcome
                "delta_hads":
                    followup_hads.iloc[0],

                # covariates
                "baseline_hads":
                    baseline_hads.iloc[0],

                "age": age,
                "gender": gender
            })

    theta_change_df = pd.DataFrame(rows)

    return theta_change_df

def run_mixed_model(theta_change_df):

    analysis_df = remove_outliers_iqr(
        theta_change_df,
        ["delta_theta", "delta_hads"]
    )


    for stage in ["early", "mid"]:

        stage_df = analysis_df[
            analysis_df["stage"] == stage
        ].copy()

        model = smf.ols(
            formula=
            "delta_hads ~ delta_theta + baseline_hads + age + C(gender)",
            data=stage_df
        ).fit()

        print(f"\n=== {stage.upper()} ===")
        print(model.summary())

def get_fo_delta(state: int):

    df_clin = load_and_prep_data(10)

    admn_df = df_clin[
        df_clin["state"].astype(int) == state
    ].copy()


    session_pairs = [
        (1, 2, "early"),
        (2, 3, "mid")
    ]

    fo_rows = []

    for pre_sess, post_sess, stage in session_pairs:

        for patient_id in admn_df["patient"].unique():

            patient_df = admn_df[
                admn_df["patient"].astype(str) == str(patient_id)
            ]

            pre_fo = patient_df.loc[
                (patient_df["session"] == pre_sess) &
                (patient_df["tms"] == "pre"),
                "fo"
            ]

            post_fo = patient_df.loc[
                (patient_df["session"] == pre_sess) &
                (patient_df["tms"] == "post"),
                "fo"
            ]

            if pre_fo.empty or post_fo.empty:
                continue

            fo_rows.append({

                "patient_id": str(patient_id),
                "stage": stage,

                "delta_fo":
                    post_fo.iloc[0] - pre_fo.iloc[0],

                "baseline_hads":
                    patient_df.loc[
                        (patient_df["session"] == pre_sess) &
                        (patient_df["tms"] == "pre"),
                        "hads_dep_total"
                    ].iloc[0],

                "followup_hads":
                    patient_df.loc[
                        (patient_df["session"] == post_sess) &
                        (patient_df["tms"] == "pre"),
                        "hads_dep_total"
                    ].iloc[0],

                "age":
                    patient_df["age"].iloc[0],

                "gender":
                    patient_df["gender"].iloc[0]
            })

    fo_change_df = pd.DataFrame(fo_rows)

    return fo_change_df


def run_mixed_model_incl_deltafo():

    analysis_df = theta_change_df.merge(
        fo_change_df,
        on=["patient_id", "stage"],
        how="inner"
    )

    for col in [
        "delta_theta",
        "delta_fo",
        "baseline_hads",
        "age"
    ]:
        analysis_df[col + "_z"] = zscore(
            analysis_df[col]
        )

    for stage in ["early", "mid"]:

        stage_df = analysis_df[
            analysis_df["stage"] == stage
        ].copy()

        model = smf.ols(
        formula=
        "followup_hads ~ delta_theta_z + delta_fo_z + "
        "baseline_hads_z + age_z + C(gender)",
        data=stage_df
    ).fit()

        print(f"\n=== {stage.upper()} ===")
        print(model.summary())



def load_and_prep_data(n_states, exclude_repeater: bool = False):

    # where are the HMM summary stats stored
    hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

    csv_path = Path(f'{hmm_dir}/hmm_demo_hads2704_{n_states}.csv')
    df = pd.read_csv(csv_path)

    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(df.patient)

    print(len(unique_ids))

    # exclude patients that repeated TMS treatment?
    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        print(f'{len(repeater_ids)} patients repeated the treatment')
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].str.contains("R")]
        drop_indices = repeater_positions
        np.save(f'{hmm_dir}/dropped_indices.npy', np.array(drop_indices))

    print(f"Analyzing {df['patient'].nunique()} patients")

    # unique patients AFTER filtering
    patient_ids = df["patient"].unique()
    print(f"Total patients: {len(patient_ids)}")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", 'responder', 'group']:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", 'responder', 'group', 'gender']:
        df[col] = df[col].astype("category", errors="ignore")

    return df_clin


def remove_outliers_iqr(df, cols):

    keep = np.ones(len(df), dtype=bool)

    for col in cols:

        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        keep &= (df[col] >= lower) & (df[col] <= upper)

    return df[keep]

