import glob
import re
import pandas as pd
from pathlib import Path
import os

# set working directory
os.chdir(Path("/home/carinaf"))

base_dir = os.getcwd()

def get_latest_file(directory, pattern):
    files = sorted(
        glob.glob(str(Path(directory) / pattern)),
        key=lambda x: Path(x).stat().st_mtime,
        reverse=True
    )
    return files[0] if files else None


def normalize_patient_id(series):
    return series.str.replace(r'^D|MB4$', '', regex=True)


def extract_scale(
    xls,
    sheet,
    item_prefix,
    new_prefix,
    additional_columns,
    total_rename=None
):
    df = xls.parse(sheet)

    # item columns sorted numerically
    item_cols = sorted(
        [c for c in df.columns if c.startswith(item_prefix)],
        key=lambda c: int(re.search(r'(\d+)', c).group())
    )

    # keep only existing additional columns
    add_cols = [c for c in additional_columns if c in df.columns]

    df = df[add_cols + item_cols].copy()

    # rename items
    df.columns = (
        df.columns
          .str.replace(rf'^{item_prefix}\s*', f'{new_prefix}_', regex=True)
    )

    # rename total column if needed
    if total_rename:
        df = df.rename(columns=total_rename)

    # normalize patient id
    df['patient'] = normalize_patient_id(df['Participant ID'])

    # drop Participant ID since we have patient
    df = df.drop(columns='Participant ID')

    return df


def combine_dataframes(base_dir: str):

    clinical_dir = Path(
        base_dir,
        "LabData",
        "Lab_LucaC",
        "A_QNC_Databank",
        "Participants_Clinical_TMS_Data"
    )

    latest_file = get_latest_file(
        clinical_dir,
        "*MDD Anonymised QNC Clinical Data.xlsx"
    )

    if latest_file is None:
        raise FileNotFoundError("No clinical Excel file found")

    print("Latest file:", latest_file)

    xls = pd.ExcelFile(latest_file)

    # -------- HADS --------
    df_hads = extract_scale(
        xls=xls,
        sheet="HADS",
        item_prefix="Item",
        new_prefix="hads",
        additional_columns=[
            'Participant ID',
            'Session',
            'Anxiety Score Total',
            'Depression Score Total',
            'Date Completed'
        ],
        total_rename={'Depression Score Total': 'hads_dep_total', 'Anxiety Score Total': 'hads_anx_total',
                     'Date Completed': 'hads_date'}
    )

    # -------- MADRS --------
    df_madrs = extract_scale(
        xls=xls,
        sheet="MADRS",
        item_prefix="Item",
        new_prefix="madrs",
        additional_columns=[
            'Participant ID',
            'Session',
            'Computation Total'
        ],
        total_rename={'Computation Total': 'madrs_total'}
    )

    # -------- HAMA --------
    df_hama = extract_scale(
        xls=xls,
        sheet="HAM-A",
        item_prefix="Item",
        new_prefix="hama",
        additional_columns=[
            'Participant ID',
            'Session',
            'Computation Total'
        ],
        total_rename={'Computation Total': 'hama_total'}
    )

    # -------- DEMOGRAPHICS --------
    df_demo = xls.parse(0)[[
        'Participant ID',
        'Age',
        'Gender',
        'Treatment Number',
        'Diagnostic Type',
        'Research Tier',
        'Age of Symptom Onset',
        'Previous TMS',
        'Previous ECT',
        'Number of Treatment Days',
        'TMS Outcome'
    ]].copy()

    df_demo['patient'] = normalize_patient_id(df_demo['Participant ID'])

    # -------- MERGING --------
    df_combined = (
        df_hads
        .merge(df_madrs, on=['patient', 'Session'], how='outer')
        .merge(df_hama, on=['patient', 'Session'], how='outer')
        .merge(df_demo.drop(columns='Participant ID'), on='patient', how='left')
    )

    df_combined['Session'] = df_combined['Session'].str.lower()

    # make sure all columns are lower case
    df_combined.columns = [c.lower() for c in df_combined.columns]

    # Move 'patient' to the first column
    cols = ['patient'] + [c for c in df_combined.columns if c != 'patient']
    df_combined = df_combined[cols]

    df_combined.to_csv(f'{base_dir}/clinical_demo_combined_012026.csv')

    return df_combined
