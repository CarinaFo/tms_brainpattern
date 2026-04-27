# core libraries
import pandas as pd
import numpy as np
from pathlib import Path

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

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

# where are the source recos stored? this is the patient ID base
prep_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_05Hz_1Hzfiltereddata')

def match_hads_to_eeg_visits():
    """
    Match weekly HADS to 3 EEG assessment visits:
      - visit 1 = first EEG pair (session 1: pre/post)
      - visit 2 = second EEG pair (session 11: pre/post)
      - visit 3 = third EEG pair (session 20: pre/post)

    Rules:
      - baseline (visit 1): use closest HADS on/before EEG date only
      - visit 2 and 3: use closest HADS in either direction
      - assign the matched HADS to both pre and post EEG recordings in that visit
      - also retain all HADS after visit 3 for later prediction analyses
    """
    import pandas as pd
    import numpy as np
    from pathlib import Path
    import re
    import glob

    # -----------------------------
    # 1) Load latest clinical file
    # -----------------------------
    clinical_dir = f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample'
    pattern = f"{clinical_dir}/*MDD Anonymised QNC Clinical Data_carinassample.xlsx"
    excel_files = glob.glob(pattern)
    excel_files = sorted(excel_files, key=lambda x: Path(x).stat().st_mtime, reverse=True)
    latest_excel = excel_files[0] if excel_files else None

    if latest_excel is None:
        raise FileNotFoundError("No clinical Excel file found.")

    print("Latest clinical file:", latest_excel)

    df_hads = pd.read_excel(latest_excel, sheet_name="HADS")
    df_hads['patient'] = df_hads['Participant ID'].astype(str).str.replace(r'^D|MB4$', '', regex=True)
    df_hads['Date Completed'] = pd.to_datetime(df_hads['Date Completed'], errors='coerce', dayfirst=True)
    df_hads = df_hads.dropna(subset=['patient', 'Date Completed']).copy()

    # Optional: standardize HADS session labels if present
    if 'Session' in df_hads.columns:
        df_hads['Session'] = df_hads['Session'].astype(str).str.lower().str.strip()

    # -----------------------------
    # 2) Load EEG files
    # -----------------------------
    eeg_dir = Path(f"{base_dir}/Lab_LucaC/A_QNC_ANT_Data/TMS_MDD_EEG_data")
    eeg_files = list(eeg_dir.rglob("*.vhdr"))

    eeg_rows = []
    for filepath in eeg_files:
        filename = filepath.name

        match_id = re.search(r'(\d{2,3}R?)', filename)
        match_datetime = re.search(r'(\d{4}-\d{2}-\d{2})[T_\-]?(\d{2}[-_]\d{2}[-_]\d{2})?', filename)

        if not (match_id and match_datetime):
            continue

        patient_id = match_id.group(1)
        if patient_id.endswith("R"):
            patient_id = patient_id[:-1].zfill(3) + "R"
        else:
            patient_id = patient_id.zfill(3)

        date_part = match_datetime.group(1)
        time_part = match_datetime.group(2) or "00-00-00"
        time_part = time_part.replace("_", "-")
        full_dt_str = f"{date_part} {time_part.replace('-', ':')}"

        try:
            session_dt = pd.to_datetime(full_dt_str)
        except Exception:
            continue

        eeg_rows.append({
            "file_path": str(filepath),
            "filename": filename,
            "patient": patient_id,
            "session_datetime": session_dt
        })

    eeg_df = pd.DataFrame(eeg_rows)
    if eeg_df.empty:
        raise ValueError("No EEG files parsed successfully.")

    eeg_df = eeg_df.sort_values(by=["patient", "session_datetime"]).copy()

    # ----------------------------------------------------
    # 3) Define 6 EEG rows as 3 visits (each visit = 2 EEG)
    # ----------------------------------------------------
    eeg_df["eeg_index"] = eeg_df.groupby("patient").cumcount()
    eeg_df["visit_number"] = eeg_df["eeg_index"] // 2 + 1
    eeg_df["tms"] = np.where(eeg_df["eeg_index"] % 2 == 0, "pre", "post")

    # Map visit_number to actual treatment session anchors
    visit_to_treatment_session = {1: 1, 2: 11, 3: 20}
    eeg_df["treatment_session"] = eeg_df["visit_number"].map(visit_to_treatment_session)

    # Keep only first 3 visits (6 EEG recordings)
    eeg_df = eeg_df[eeg_df["visit_number"].isin([1, 2, 3])].copy()

    # ----------------------------------------------------------
    # 4) Collapse each pre/post pair into one EEG visit anchor
    # ----------------------------------------------------------
    # Use the PRE EEG as the anchor if present, otherwise earliest in that visit
    visit_rows = []
    for (patient, visit_number), grp in eeg_df.groupby(["patient", "visit_number"]):
        grp = grp.sort_values("session_datetime").copy()

        pre_rows = grp[grp["tms"] == "pre"]
        if not pre_rows.empty:
            anchor_dt = pre_rows.iloc[0]["session_datetime"]
        else:
            anchor_dt = grp.iloc[0]["session_datetime"]

        visit_rows.append({
            "patient": patient,
            "visit_number": visit_number,
            "treatment_session": grp["treatment_session"].iloc[0],
            "visit_anchor_datetime": anchor_dt
        })

    visit_df = pd.DataFrame(visit_rows)

    # ----------------------------------------------------------
    # 5) Match one HADS row to each EEG visit
    # ----------------------------------------------------------
    matched_visit_rows = []
    future_hads_rows = []

    for _, visit_row in visit_df.iterrows():
        patient = visit_row["patient"]
        visit_number = visit_row["visit_number"]
        treatment_session = visit_row["treatment_session"]
        eeg_dt = visit_row["visit_anchor_datetime"]

        patient_hads = df_hads[df_hads["patient"] == patient].copy()
        if patient_hads.empty:
            continue

        patient_hads["days_from_eeg"] = (
            patient_hads["Date Completed"] - eeg_dt
        ).dt.total_seconds() / (60 * 60 * 24)

        # Visit 1: baseline must be on/before EEG date
        if visit_number == 1:
            eligible = patient_hads[patient_hads["days_from_eeg"] <= 0].copy()
            if eligible.empty:
                closest = None
            else:
                # closest prior = largest negative / zero
                closest = eligible.loc[eligible["days_from_eeg"].idxmax()]
        else:
            # Visits 2 and 3: closest in absolute time
            patient_hads["abs_days_from_eeg"] = patient_hads["days_from_eeg"].abs()
            closest = patient_hads.loc[patient_hads["abs_days_from_eeg"].idxmin()]

        if closest is not None:
            matched_visit_rows.append({
                "patient": patient,
                "visit_number": visit_number,
                "treatment_session": treatment_session,
                "visit_anchor_datetime": eeg_dt,
                "matched_hads_date": closest["Date Completed"],
                "matched_hads_session": closest["Session"] if "Session" in closest else np.nan,
                "matched_anxiety": closest["Anxiety Score Total"] if "Anxiety Score Total" in closest else np.nan,
                "matched_depression": closest["Depression Score Total"] if "Depression Score Total" in closest else np.nan,
                "matched_days_diff": closest["days_from_eeg"],
            })

        # After visit 3, retain ALL later HADS for future prediction analyses
        if visit_number == 3:
            future_hads = patient_hads[patient_hads["Date Completed"] > eeg_dt].copy()
            if not future_hads.empty:
                future_hads = future_hads.sort_values("Date Completed")
                for _, hads_row in future_hads.iterrows():
                    future_hads_rows.append({
                        "patient": patient,
                        "final_visit_number": 3,
                        "final_treatment_session": treatment_session,
                        "final_eeg_datetime": eeg_dt,
                        "future_hads_date": hads_row["Date Completed"],
                        "future_hads_session": hads_row["Session"] if "Session" in hads_row else np.nan,
                        "future_anxiety": hads_row["Anxiety Score Total"] if "Anxiety Score Total" in hads_row else np.nan,
                        "future_depression": hads_row["Depression Score Total"] if "Depression Score Total" in hads_row else np.nan,
                        "days_after_final_eeg": (
                            hads_row["Date Completed"] - eeg_dt
                        ).total_seconds() / (60 * 60 * 24),
                    })

    matched_visit_df = pd.DataFrame(matched_visit_rows)
    future_hads_df = pd.DataFrame(future_hads_rows)

    # -----------------------------------------------------------------
    # 6) Expand visit-level HADS match back onto the 6 individual EEG rows
    # -----------------------------------------------------------------
    eeg_matched_df = pd.merge(
        eeg_df,
        matched_visit_df,
        on=["patient", "visit_number", "treatment_session"],
        how="left"
    )

    # Save outputs
    eeg_matched_df.to_csv(f"{hmm_dir}/hads_matched_to_eeg_visits.csv", index=False)
    matched_visit_df.to_csv(f"{hmm_dir}/hads_matched_visit_level.csv", index=False)
    future_hads_df.to_csv(f"{hmm_dir}/hads_after_final_eeg.csv", index=False)

    return eeg_matched_df, matched_visit_df, future_hads_df






def save_summary_stats_df(n_sessions: int = 6, n_states: int = None):
    """
    Build merged EEG + HMM + clinical dataframe.

    Uses:
      - visit-level HADS matching from match_hads_to_eeg_visits()
      - 6 EEG recordings per patient
      - 3 visits total:
            visit 1 -> treatment session 1
            visit 2 -> treatment session 11
            visit 3 -> treatment session 20
      - each visit has pre/post EEG
    """

    # ---------------------------------------------------------
    # 1) Match HADS to EEG visits
    # ---------------------------------------------------------
    eeg_matched_df, matched_visit_df, future_hads_df = match_hads_to_eeg_visits()

    # eeg_matched_df should contain one row per EEG file:
    # patient, visit_number, treatment_session, tms, matched_depression, etc.

    # ---------------------------------------------------------
    # 2) Load combined clinical/demographic dataframe
    # ---------------------------------------------------------
    df = pd.read_csv(
        f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/clinical_demo_combined_270426.csv'
    )

    # standardize columns just in case
    df.columns = (
        df.columns.str.replace('.', '_', regex=False)
                  .str.lower()
                  .str.strip()
    )

    # ---------------------------------------------------------
    # 3) Load patient IDs included in HMM
    # ---------------------------------------------------------
    ids_list = pd.read_csv(f'{prep_dir}/patients_fitted_for_this_hmm.csv')
    ids_list = pd.unique(ids_list['patient_id'])

    # keep only patients present in HMM
    eeg_matched_df = eeg_matched_df[eeg_matched_df['patient'].isin(ids_list)].copy()
    df = df[df['patient'].isin(ids_list)].copy()

    # ---------------------------------------------------------
    # 4) Build HMM dataframe from saved arrays
    # ---------------------------------------------------------
    rows = []

    for session_idx in range(n_sessions):
        fo = np.load(f"{hmm_dir}/fo_{session_idx}_{n_states}.npy")
        lt = np.load(f"{hmm_dir}/lt_{session_idx}_{n_states}.npy")
        intv = np.load(f"{hmm_dir}/intv_{session_idx}_{n_states}.npy")
        sr = np.load(f"{hmm_dir}/sr_{session_idx}_{n_states}.npy")

        # session_idx: 0..5 -> 3 visits x pre/post
        visit_number = (session_idx // 2) + 1
        tms = 'pre' if session_idx % 2 == 0 else 'post'

        # map visit to actual treatment session anchor
        visit_to_treatment_session = {1: 1, 2: 11, 3: 20}
        treatment_session = visit_to_treatment_session.get(visit_number)

        for idx, patient in enumerate(ids_list):
            for state in range(fo.shape[1]):
                rows.append({
                    'patient': patient,
                    'visit_number': visit_number,
                    'treatment_session': treatment_session,
                    'tms': tms,
                    'state': state,
                    'fo': fo[idx, state],
                    'lt': lt[idx, state],
                    'intv': intv[idx, state],
                    'sr': sr[idx, state]
                })

    df_hmm = pd.DataFrame(rows)

    # ---------------------------------------------------------
    # 5) Merge EEG/HADS matches with HMM features
    # ---------------------------------------------------------
    # This is the main new merge:
    # join on patient + visit_number + tms
    df_eeg = pd.merge(
        eeg_matched_df,
        df_hmm,
        on=['patient', 'visit_number', 'tms', 'treatment_session'],
        how='inner'
    )

    # ---------------------------------------------------------
    # 6) Prepare clinical/demographic table for merging
    # ---------------------------------------------------------
    # We want clinical variables that are stable or tied to treatment status.
    # We do NOT want to duplicate old HADS session-matching logic.
    #
    # Since your EEG visits correspond to treatment sessions 1, 11, 20,
    # we map those values onto the clinical dataframe if needed.

    clinical_df = df.copy()

    # standardize session column if present
    if 'session' in clinical_df.columns:
        clinical_df['session'] = clinical_df['session'].astype(str).str.lower().str.strip()

        # map pre/post rows from the clinical table if applicable
        session_map = {'pre': 1, 'post': 3}
        if clinical_df['session'].isin(session_map.keys()).any():
            clinical_df['clinical_session_code'] = clinical_df['session'].map(session_map)

    # Keep one row per patient where possible for demographics
    # If your file has repeated rows per patient, select useful columns and deduplicate.
    demographic_cols = [
        c for c in clinical_df.columns
        if c in [
            'patient',
            'age',
            'gender',
            'diagnostic type',
            'age of symptom onset',
            'tms outcome'
        ]
    ]

    if len(demographic_cols) > 1:
        df_demo = clinical_df[demographic_cols].drop_duplicates(subset=['patient'])
        df_eeg = pd.merge(df_eeg, df_demo, on='patient', how='left')

    # ---------------------------------------------------------
    # 7) Clean columns
    # ---------------------------------------------------------
    df_eeg = df_eeg.loc[:, ~df_eeg.columns.str.contains('^unnamed', case=False)]
    df_eeg = df_eeg.dropna(axis=1, how='all')

    df_eeg.columns = (
        df_eeg.columns.str.replace('.', '_', regex=False)
                        .str.lower()
                        .str.strip()
    )

    df_eeg = df_eeg.loc[:, ~df_eeg.columns.str.endswith(('_x', '_y'))]

    # ---------------------------------------------------------
    # 8) Derived variables
    # ---------------------------------------------------------
    if 'tms outcome' in df_eeg.columns:
        df_eeg['responder'] = np.where(
            df_eeg['tms outcome'].isin([1, 2]), 0,
            np.where(df_eeg['tms outcome'].isin([3, 4]), 1, np.nan)
        )

    if 'diagnostic type' in df_eeg.columns:
        df_eeg['group'] = df_eeg['diagnostic type'].map({1: 1, 2: 2, 3: 3, 4: 3})

    # ---------------------------------------------------------
    # 9) Add visit-level symptom change scores
    # ---------------------------------------------------------
    # Compute from matched HADS-D at the visit level, then merge back.
    if not matched_visit_df.empty:
        hads_change = matched_visit_df.pivot(
            index='patient',
            columns='visit_number',
            values='matched_depression'
        ).reset_index()

        # rename columns if present
        rename_map = {}
        if 1 in hads_change.columns:
            rename_map[1] = 'hads_d_visit1'
        if 2 in hads_change.columns:
            rename_map[2] = 'hads_d_visit2'
        if 3 in hads_change.columns:
            rename_map[3] = 'hads_d_visit3'
        hads_change = hads_change.rename(columns=rename_map)

        if {'hads_d_visit1', 'hads_d_visit2'}.issubset(hads_change.columns):
            hads_change['delta_hads_d_1_to_2'] = (
                hads_change['hads_d_visit2'] - hads_change['hads_d_visit1']
            )

        if {'hads_d_visit2', 'hads_d_visit3'}.issubset(hads_change.columns):
            hads_change['delta_hads_d_2_to_3'] = (
                hads_change['hads_d_visit3'] - hads_change['hads_d_visit2']
            )

        if {'hads_d_visit1', 'hads_d_visit3'}.issubset(hads_change.columns):
            hads_change['delta_hads_d_1_to_3'] = (
                hads_change['hads_d_visit3'] - hads_change['hads_d_visit1']
            )

        df_eeg = pd.merge(df_eeg, hads_change, on='patient', how='left')
    
    df_eeg = df_eeg.rename(columns={"matched_depression": "hads_dep_total"})
    df_eeg = df_eeg.rename(columns={"visit_number": "session"})

    # ---------------------------------------------------------
    # 10) Save outputs
    # ---------------------------------------------------------
    df_eeg.to_csv(Path(f"{hmm_dir}/hmm_demo_hads2704_{n_states}.csv"), index=False)

    if not future_hads_df.empty:
        future_hads_df.to_csv(
            Path(f"{hmm_dir}/future_hads_after_final_eeg_{n_states}.csv"),
            index=False
        )

    return df_eeg

if __name__ == "__main__":
    for st in [6, 8, 10, 12]:
        save_summary_stats_df(6, st)