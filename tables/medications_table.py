# -*- coding: utf-8 -*-
"""
Medication tables for supplementary figures: Current and previous medications
Author: Carina Forster
"""

import pandas as pd
import re
from pathlib import Path
import matplotlib.pyplot as plt

# ---------------------------
# Plot settings
# ---------------------------
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

# ---------------------------
# Constants
# ---------------------------
NO_INFO_LABEL = "No medication information"
UNCLASSIFIED_LABEL = "Unclassified"

# ---------------------------
# Cleaning functions
# ---------------------------
def clean_participant_id(x):
    if pd.isna(x):
        return pd.NA
    match = re.search(r"(\d{3}R?)", str(x))
    return match.group(1) if match else pd.NA

def clean_text(x):
    x = str(x).lower()
    # Remove punctuation
    x = re.sub(r"[^a-z0-9\s]", " ", x)
    # Normalize spaces
    x = re.sub(r"\s+", " ", x)
    # Remove combined dosage like 60mg, 10 mcg, 5 ml
    x = re.sub(r"\b\d+(\.\d+)?\s*(mg|mcg|g|ml|iu)\b", " ", x)
    # Remove remaining numbers
    x = re.sub(r"\b\d+(\.\d+)?\b", "", x)
    # Remove frequency words
    x = re.sub(r"\b(daily|once|twice|bid|tid|qid|qd|od|nocte|weekly|monthly)\b", " ", x)
    # Remove random words
    x = re.sub(r"\b(when|la|required)\b", " ", x)
    # Normalize spaces again
    x = re.sub(r"\s+", " ", x)
    return x.strip()


def clean_and_split_meds(x, missing={"", "na", "n/a", "nan", "none", "unknown"}):
    if pd.isna(x) or x.lower() in missing:
        return pd.NA
    x = clean_text(x)
    meds = re.split(r",|;| and |\s+", x)
    meds = [m for m in meds if m]
    return meds

# ---------------------------
# Load data
# ---------------------------
def load_and_clean_data():
    # Participant info
    df = pd.read_excel(
        Path("L:/Lab_LucaC/A_QNC_Databank/Participants_Clinical_TMS_Data/19012026 MDD Anonymised QNC Clinical Data.xlsx"),
        sheet_name='Participant Info'
    )

    # Medication classes
    med_classes_df = pd.read_excel(
        "L:/Lab_LucaC/Carina/canonical_hmm_finalsample/medication_classes_bjorn.xlsx"
    )

    # Keep relevant columns
    df = df[[
        "Participant ID", "Comorbidities",
        "Current Medication and Dose",
        "Previous Antidepressants or Augmentations"
    ]]

    # Clean participant IDs
    df['patient'] = df['Participant ID'].apply(clean_participant_id)

    # Filter to included patients
    included_ids = pd.read_csv(
        Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_05Hz_1Hzfiltereddata/patients_fitted_for_this_hmm.csv")
    )["patient_id"]
    df = df[df['patient'].isin(included_ids)].copy()

    # Drop original ID column
    df.drop('Participant ID', axis=1, inplace=True)

    # Clean meds columns
    df["current_meds"] = df["Current Medication and Dose"].apply(clean_and_split_meds)
    df["previous_meds"] = df["Previous Antidepressants or Augmentations"].apply(clean_and_split_meds)

    return df, med_classes_df

df, med_classes_df = load_and_clean_data()

def get_medication_tables():
    # ---------------------------
    # Build medication lookup table
    # ---------------------------
    med_class_lookup = {}
    for cls in med_classes_df.columns:
        meds = (
            med_classes_df[cls]
            .dropna()
            .astype(str)
            .str.lower()
            .str.strip()
        )
        for med in meds:
            med_class_lookup[med] = cls

    # ---------------------------
    # Map medications to classes
    # ---------------------------
    def meds_to_classes(med_list, lookup):
        """Map a patient’s meds to classes, handle unknown/missing."""
        if med_list is pd.NA or med_list is None:
            return [NO_INFO_LABEL]

        classes = set()
        for med in med_list:
            if med in lookup:
                classes.add(lookup[med])
        if len(classes) == 0:
            return [UNCLASSIFIED_LABEL]
        return list(classes)

    df["previous_med_classes"] = df["previous_meds"].apply(lambda x: meds_to_classes(x, med_class_lookup))
    df["current_med_classes"] = df["current_meds"].apply(lambda x: meds_to_classes(x, med_class_lookup))

    # ---------------------------
    # Build publication table
    # ---------------------------
    def medication_class_table(df, class_col):
        """Return table of n (%) patients per medication class."""
        n_patients = df['patient'].nunique()

        counts = (
            df[['patient', class_col]]
            .explode(class_col)
            .groupby(class_col)['patient']
            .nunique()
            .reset_index(name='n_patients')
        )

        counts['percent'] = (100 * counts['n_patients'] / n_patients).round(1)
        counts['n (%)'] = counts['n_patients'].astype(str) + ' (' + counts['percent'].astype(str) + '%)'

        # Optional: move missing info to bottom
        counts['sort_key'] = counts['Medication class'] = counts[class_col].eq(NO_INFO_LABEL)
        counts = counts.sort_values('sort_key').drop(columns='sort_key')

        return counts.rename(columns={class_col: 'Medication class'})[['Medication class', 'n (%)']]

    prev_table = medication_class_table(df, 'previous_med_classes')
    curr_table = medication_class_table(df, 'current_med_classes')

    curr_table.to_csv('current_medications_grouped.csv')
    prev_table.to_csv('previous_medications_grouped.csv')


def get_table_comorbidities():
    
    s = (
    df['Comorbidities']
      .astype(str)          # ensures everything is a string
      .str.strip()          # removes whitespace-only rows
    )
    # empty rows are now NaN

    # 1) Split into list
    df = df.copy()
    df['comorb_clean'] = (
        s.str.lower()
        .str.replace(r'[;/|]', ',', regex=True)   # add | as separator too
        .str.replace(r'\s+', ' ', regex=True)     # normalise weird spacing
        .str.split(',')
    )

    df = df.explode('comorb_clean')

    # 5 patients have no comorbidity infos
    df = df[df['comorb_clean'] != 'nan']

    # 2) Mapping dictionary
    comorb_mapping = {
        r'\b(gad|generalized anxiety|generalised anxiety)\b':
            'Generalised anxiety disorder',
        r'\b(ptsd|post[- ]?traumatic stress)\b':
            'Post-traumatic stress disorder',
        r'\b(adhd|add|attention deficit)\b':
            'Attention-deficit/hyperactivity disorder',
        r'\b(social anxiety|social phobia|sociel anxiety)\b':
            'Social anxiety disorder',
        r'\b(ocd|obsessive compulsive)\b':
            'Obsessive-compulsive disorder',
        r'\b(asd|autism|autistic)\b':
            'Autism spectrum disorder',
        r'\b(tourette|tourettes|ts)\b':
            'Tourette syndrome',
        r'\b(bulimia|bulimia nervosa|anorexia|anorexia nervosa|eating disorder nos|binge eating disorder)\b':
            'Eating disorder',
        r'\b(alcohol misuse|alcohol abuse|alcohol use disorder|aud|alcohol miuse disorder)\b':
            'Alcohol use disorder',
        r'\b(panic disorder)\b':
            'Panic disorder',
        r'\b(borderline personality disorder)\b':
            'Borderline personality disorder',    
    }

    def map_comorb_all(x: str):
        hits = []
        for pattern, name in comorb_mapping.items():
            if re.search(pattern, x):
                hits.append(name)
        return hits if hits else None

    df['comorb_std'] = df['comorb_clean'].apply(map_comorb_all)

    # explode the list of matches so each patient can have multiple diagnoses
    df2 = df.explode('comorb_std').dropna(subset=['comorb_std'])

    # patient-level counts (each patient counts once per comorbidity)
    table = (
        df2[['patient', 'comorb_std']]
        .drop_duplicates()
        .value_counts('comorb_std')
        .reset_index(name='n')
    )

    denom = df['patient'].nunique()  # or original df if you prefer
    table['percent'] = table['n'] / denom * 100
    table = table.sort_values(['n', 'comorb_std'], ascending=[False, True])

    table.to_csv('comorbidites_grouped.csv')

