# core libraries
import pandas as pd
import numpy as np
from pathlib import Path
import os
import re
import glob

# run in base python (3.12)

system='linux'

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/LabData')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:")
else:
    "No available system path defined *windows* or *linux*"

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

# where are the source recos stored? this is the patient ID base
prep_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_05Hz_1Hzfiltereddata')

for st in [6, 8, 10]:
    save_summary_stats_df(6, st)

def match_hads_to_eeg_session():
    """we know the date but not the time of the hads conducted, so we can match
    hads to eeg recordings based on data"""

    # Define the directory where the Excel files are stored
    clinical_dir = f'{base_dir}/Lab_LucaC/A_QNC_Databank/Participants_Clinical_TMS_Data'

    # Pattern: anything before "Anonymised QNC Clinical Data.csv"
    pattern = f"{clinical_dir}/*MDD Anonymised QNC Clinical Data.xlsx"

    # Get all CSV files in that directory
    csv_files = glob.glob(pattern)

    # Sort by modification time (latest first)
    csv_files = sorted(csv_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)

    # Get the latest file if it exists
    latest_csv = csv_files[0] if csv_files else None

    print("Latest CSV file:", latest_csv)

    # Load HADS data
    df_hads = pd.read_excel(latest_csv, sheet_name="HADS")
    df_hads['patient'] = df_hads['Participant ID'].str.replace(r'^D|MB4$', '', regex=True)
    df_hads['Date Completed'] = pd.to_datetime(df_hads['Date Completed'], errors='coerce', dayfirst=True)
    df_hads = df_hads.dropna(subset=['Date Completed'])

    # Load EEG data file paths
    eeg_dir = Path(f"{base_dir}/Lab_LucaC/A_QNC_ANT_Data/TMS_MDD_EEG_data")
    eeg_files = list(eeg_dir.rglob("*.vhdr"))

    eeg_data = []
    for filepath in eeg_files:
        filename = filepath.name
        match_id = re.search(r'(\d{2,3}R?)', filename)
        # Extract timestamp (datetime format like 2023-07-12_14-32-00)
        match_datetime = re.search(r'(\d{4}-\d{2}-\d{2})[T_\-]?(\d{2}[-_]\d{2}[-_]\d{2})?', filename)


        if match_id and match_datetime:
            patient_id = match_id.group(1)
            # Pad to 3 digits if necessary (preserve 'R')
            if patient_id.endswith("R"):
                patient_id = patient_id[:-1].zfill(3) + "R"
            else:
                patient_id = patient_id.zfill(3)
            date_part = match_datetime.group(1)
            time_part = match_datetime.group(2) or "00-00-00"

            # Normalize to full datetime
            time_part = time_part.replace("_", "-")  # just in case
            full_dt_str = f"{date_part} {time_part.replace('-', ':')}"
            try:
                session_dt = pd.to_datetime(full_dt_str)
            except Exception:
                continue

            eeg_data.append({
                "file_path": filepath,
                "filename": filename,
                "patient": patient_id,
                "session_datetime": session_dt
            })

    # store in dataframe
    eeg_df = pd.DataFrame(eeg_data)

    # we sort the sessions ascending
    eeg_df = eeg_df.sort_values(by=["patient", "session_datetime"]).copy()

    # Add session and TMS column for each patient
    eeg_df['session_number'] = eeg_df.groupby("patient").cumcount() // 2 + 1
    eeg_df['tms'] = eeg_df.groupby("patient").cumcount() % 2
    eeg_df['tms'] = eeg_df['tms'].map({0: 'pre', 1: 'post'})

    # Match EEG recordings to the closest previous HADS score
    matched_rows = []

    for _, eeg_row in eeg_df.iterrows():
        patient = eeg_row['patient']
        eeg_dt = eeg_row['session_datetime']

        # Filter HADS scores for this patient
        patient_hads = df_hads[df_hads['patient'] == patient].copy()

        # skip if we don't have hads score
        if patient_hads.empty:
            continue

        # Calculate time difference (positive for future, negative for past)
        patient_hads['time_diff_days'] = (patient_hads['Date Completed'] - eeg_dt).dt.total_seconds() / (60 * 60 * 24)

        # Prioritize scores before the EEG
        pre_hads = patient_hads[patient_hads['time_diff_days'] <= 0]

        if not pre_hads.empty:
            # Take the HADS closest before the EEG
            closest = pre_hads.loc[pre_hads['time_diff_days'].idxmax()]
        else:
            # If none before, take the soonest after
            post_hads = patient_hads[patient_hads['time_diff_days'] > 0]
            if post_hads.empty:
                continue
            closest = post_hads.loc[post_hads['time_diff_days'].idxmin()]

        matched = eeg_row.to_dict()
        matched.update({
            "matched_hads_date": closest['Date Completed'],
            "matched_hads_session": closest['Session'],
            "matched_anxiety": closest['Anxiety Score Total'],
            "matched_depression": closest['Depression Score Total']
        })
        matched_rows.append(matched)

    matched_df = pd.DataFrame(matched_rows)

    matched_df['matched_hads_session'] = matched_df['matched_hads_session'].str.lower()

    matched_df.to_csv(f'{hmm_dir}/hads_matched_to_eeg_upto{pd.unique(matched_df.patient)[-1]}.csv')

    return matched_df


def save_summary_stats_df(n_sessions: int = 6, n_states: int = None):

    matched_df = match_hads_to_eeg_session()

    # load dataframe (contains hads, madrs, hama including subscales and demographics)
    df = pd.read_csv(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/clinical_demo_combined_012026.csv')

    # HADS scores we want to match to EEG sessions
    hads_cols = [c for c in df.columns if 'hads' in c.lower()]
    other_cols = [c for c in df.columns if c not in hads_cols]

    df_hads = df[['patient', 'session'] + hads_cols]
    df_hads_matched = pd.merge(
        df_hads,
        matched_df,
        left_on=['patient', 'session'],
        right_on=['patient', 'matched_hads_session'],
        how='right'  # keep only EEG-matched HADS
    )
    df_hads_matched = df_hads_matched.drop(columns=['session', 'matched_hads_session'])

    df_madrs_hama = df[other_cols]

    # Filter for only 'Pre' and 'Post' sessions
    df_madrs_hama = df_madrs_hama[df_madrs_hama['session'].isin(['pre', 'post'])].copy()

    # Replace 'Pre' with 1 and 'Post' with 2 in the session column
    df_madrs_hama['session'] = df_madrs_hama['session'].map({'pre': 1, 'post': 3})

    df_clinical = pd.merge(
        df_hads_matched,
        df_madrs_hama,
        left_on=['patient', 'session_number'],
        right_on=['patient', 'session'],
        how='outer'  # keep only EEG-matched HADS
    )

    ids_list = pd.read_csv(f'{prep_dir}/patients_fitted_for_this_hmm.csv')

    ids_list = pd.unique(ids_list['patient_id'])

    # filter dataframe for patients in EEG list
    df_demo = df_clinical[df_clinical['patient'].isin(ids_list)]

    rows = []

    # load all datasets
    for session_idx in range(n_sessions):
        # Load HMM feature arrays
        fo = np.load(f"{hmm_dir}/fo_{session_idx}_{n_states}.npy")  # shape (n_patients, n_states)
        lt = np.load(f"{hmm_dir}/lt_{session_idx}_{n_states}.npy")
        intv = np.load(f"{hmm_dir}/intv_{session_idx}_{n_states}.npy")
        sr = np.load(f"{hmm_dir}/sr_{session_idx}_{n_states}.npy")
        
        # 3 sessions
        session = (session_idx // 2) + 1
        # pre and post for each session
        tms = 'pre' if session_idx % 2 == 0 else 'post'
        
        for idx, patient in enumerate(ids_list):
            for state in range(fo.shape[1]):
                rows.append({
                    'patient': patient,
                    'session': session,
                    'tms': tms,
                    'state': state,
                    'fo': fo[idx, state],
                    'lt': lt[idx, state],
                    'intv': intv[idx, state],
                    'sr': sr[idx, state]
                })

    # convert to dataframe
    df_hmm = pd.DataFrame(rows)

    df_eeg = pd.merge(
        df_demo,
        df_hmm,
        left_on=['patient', 'session_number', 'tms'],
        right_on=['patient', 'session', 'tms'],
        how='inner'
    )

    df_eeg = df_eeg.rename(columns={'session_y': 'session'})

    # Drop columns that start with 'Unnamed' or are fully empty
    df_eeg = df_eeg.loc[:, ~df_eeg.columns.str.contains('^Unnamed')]
    df_eeg = df_eeg.dropna(axis=1, how='all')

    # Replace dots with underscores, lowercase, and strip whitespace
    df_eeg.columns = (
        df_eeg.columns.str.replace('.', '_', regex=False)
                .str.lower()
                .str.strip()
    )

    # Drop all columns ending with _x or _y
    df_eeg = df_eeg.loc[:, ~df_eeg.columns.str.endswith(('_x', '_y'))]

    # create new column
    df_eeg['years_with_depression'] = df_eeg['age'] - df_eeg['age of symptom onset']

    # Create the response column
    df_eeg['responder'] = np.where(
        df_eeg['tms outcome'].isin([1, 2]), 0,
        np.where(df_eeg['tms outcome'].isin([3, 4]), 1, np.nan)
    )

    df_eeg['group'] = df_eeg['diagnostic type'].map({1: 1, 2: 2, 3: 3, 4: 3})

    df_eeg.to_csv(Path(f"{prep_dir}/hmm_demo_quest_{n_states}.csv"), index=False)
    
    return "saved big dataframe"