# Beta power from ACC and symptom improvement
import os
import pandas as pd
from glob import glob
import mne
import numpy as np
from pathlib import Path
import scipy.stats as stats
import statsmodels.formula.api as smf

# Beta band EEG findings converged across trials: 
# frontal beta power decreased significantly following active but not sham SNT.
# Additionally, beta baseline activity and post-SNT changes related to treatment efficacy in the current study.
# Specifically, greater post-SNT reduction in left anterior cingulate cortex (L-ACC) beta power correlated with 
# greater clinical improvement immediately (rho=0.48, p=0.019) and 1-month after (rho=0.51, p=0.012) active SNT.
#  Moreover, higher pre-treatment L-ACC beta power predicted greater subsequent clinical benefit from active SNT
# (immediate-post: β=-10.26, p=0.0042; 1-month after: β=-9.00, p=0.024). 

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

frontal_parcels = {
    23: "L VLPFC",
    24: "R VLPFC",
    27: "L premotor",
    28: "R premotor",
    29: "L DLPFC",
    30: "R DLPFC",
    31: "L OFC",
    32: "R OFC",
    35: "L dorsal OFC",
    36: "R dorsal OFC",
    38: "ACC"
}

# windows paths
preproc_path = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/preprocessed")
source_path  = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/source_reco_giles_parcel")

# load source data
patient_sessions = {}

for sess_dir in sorted(glob(os.path.join(preproc_path, "*_*")))[:-1]:

    sess_id = os.path.basename(sess_dir)

    patient_id = sess_id.split("_")[0]

    # Source parcel data
    src_path = os.path.join(
        source_path, sess_id, "parc", "lcmv-parc-raw.fif"
    )
    if not os.path.exists(src_path):
        continue

    patient_sessions.setdefault(patient_id, []).append({
        "session": sess_id,
        "src_path": src_path
    })


df_diff = load_and_prep_data(10, False)

# load patient ID list
patient_ids = pd.read_csv(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"))

fmin=15
fmax=25

for parcelnum in frontal_parcels.keys():
    all_pre = []
    all_post = []

    for idx, patient_id in enumerate(patient_ids['patient_id']):

        print(f"\n=== Patient {patient_id} ===")

        for sess in patient_sessions[patient_id]:
            if sess['session'][-1] == '2':
                raw_src = mne.io.read_raw_fif(sess["src_path"], preload=True, verbose=False)
                acc_only = raw_src.pick_channels([f'parcel_{parcelnum}']) # ACC parcel
                psd = acc_only.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose="ERROR", picks='misc')
                beta_pow = np.mean(psd.data)
                all_pre.append(beta_pow)
            elif sess['session'][-1] == '5':
                raw_src = mne.io.read_raw_fif(sess["src_path"], preload=True, verbose=False)
                acc_only = raw_src.pick_channels([f'parcel_{parcelnum}']) # ACC parcel
                psd = acc_only.compute_psd(method="welch", fmin=fmin, fmax=fmax, verbose="ERROR", picks='misc')
                beta_pow = np.mean(psd.data)
                all_post.append(beta_pow)
            else:
                continue

    diff_beta = np.array(all_post) - np.array(all_pre)

    r, p = stats.pearsonr(diff_beta, df_diff['madrs_total_diff_s3_s1'])
    print('does beta power change correlate with symptom change')
    print(f"r = {r:.3f}, p = {p:.3f}")
    
    t, p = stats.ttest_rel(all_pre, all_post)
    print('does beta power change over treatment')
    print(f"t = {t:.3f}, p = {p:.3f}")

    df_reg = pd.DataFrame()

    df_reg['diff_hads'] = df_diff['madrs_total_diff_s3_s1']
    df_reg['beta_pre'] = all_pre

    model = smf.ols(
        "diff_hads ~ beta_pre",
        data=df_reg
    ).fit()

    print(model.summary())


def load_and_prep_data(n_states, exclude_repeater: bool = False):

    # where are the HMM summary stats stored
    hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

    csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

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
    for col in ["age", "gender", 'responder', 'group', 'years_with_depression']:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", 'responder', 'group', 'gender']:
        df[col] = df[col].astype("category", errors="ignore")

    df_pre = df[df['tms'] == 'pre']

    metrics = ["hads_dep_total", "madrs_total"]  # add others if needed

    df_wide = (
        df_pre
        .query("session in [1, 3]")
        .pivot_table(
            index=["patient"],
            columns="session",
            values=metrics
        )
    )

    df_diff = df_wide.copy()

    for metric in metrics:
        df_diff[(metric, "diff_s3_s1")] = df_wide[(metric, 3)] - df_wide[(metric, 1)]

    df_diff = df_diff.reset_index()

    # Flatten column names
    df_diff.columns = [
        f"{a}_{b}" if isinstance(a, str) else a
        for a, b in df_diff.columns
    ]

    return df_diff
