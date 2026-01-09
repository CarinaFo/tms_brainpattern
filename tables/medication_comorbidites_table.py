import pandas as pd
import re
from pathlib import Path

df = pd.read_excel(Path("L:/Lab_LucaC/A_QNC_Databank/Participants_Clinical_TMS_Data/12122025 MDD Anonymised QNC Clinical Data.xlsx"), 
                        sheet_name='Participant Information')

# Example column names (adjust to yours)
df = df[["Participant ID", "Comorbidities", "Current Medication and Dose"]]

df['patient'] = df['Participant ID'].apply(clean_participant_id)

patients_included = pd.read_csv(r"L:\Lab_LucaC\Carina\canonical_hmm_finalsample\prepared_data_giles_05Hz_1Hzfiltereddata\patients_fitted_for_this_hmm.csv")

included_ids = patients_included["patient_id"]

df = df[df['patient'].isin(included_ids)]

df.drop('Participant ID', axis=1)

df["comorbidities_clean"] = df["Comorbidities"].apply(clean_text)
df["medications_clean"] = df["Current Medication and Dose"].apply(clean_text)

df["medication_list"] = df["medications_clean"].apply(extract_medications)

med_long = (
    df[["patient", "medication_list"]]
    .explode("medication_list")
)

MEDICATION_CLASSES = {
    # SSRIs
    "Sertraline": "SSRI",
    "Fluoxetine": "SSRI",
    "Escitalopram": "SSRI",
    "Citalopram": "SSRI",
    "Paroxetine": "SSRI",

    # SNRIs
    "Venlafaxine": "SNRI",
    "Duloxetine": "SNRI",
    "Desvenlafaxine": "SNRI",

    # TCAs
    "Amitriptyline": "TCA",
    "Nortriptyline": "TCA",

    # Atypical antidepressants
    "Bupropion": "Atypical antidepressant",
    "Mirtazapine": "Atypical antidepressant",

    # Antipsychotics
    "Quetiapine": "Antipsychotic",
    "Aripiprazole": "Antipsychotic",
    "Olanzapine": "Antipsychotic",
    "Risperidone": "Antipsychotic",

    # Stimulants
    "Methylphenidate": "Stimulant",
    "Dexamphetamine": "Stimulant",
    "Lisdexamfetamine": "Stimulant",

    # Mood stabilisers
    "Lithium": "Mood stabiliser",
    "Lamotrigine": "Mood stabiliser",
    "Valproate": "Mood stabiliser",

    # Anxiolytics / sedatives
    "Diazepam": "Benzodiazepine",
    "Lorazepam": "Benzodiazepine",
    "Clonazepam": "Benzodiazepine",
    "Zolpidem": "Hypnotic",
}

# Normalize known brand names → generic
brand_to_generic = {
    "Ritalin": "Methylphenidate",
    "Ritalin La": "Methylphenidate",
    "Concerta": "Methylphenidate",
    "Prozac": "Fluoxetine",
    "Zoloft": "Sertraline",
}

med_long["Medication"] = (
    med_long["medication_list"]
    .replace(brand_to_generic)
)

med_long["Medication_Class"] = (
    med_long["Medication"]
    .map(MEDICATION_CLASSES)
    .fillna("Other / Unclassified")
)

table_class = (
    med_long
    .groupby("Medication_Class")["patient"]
    .nunique()
    .reset_index()
    .rename(columns={"patient": "N_patients"})
    .sort_values("N_patients", ascending=False)
)

n_total = med_long["patient"].nunique()

table_class["Percent"] = (
    100 * table_class["N_patients"] / n_total
).round(1)

table_drug = (
    med_long
    .groupby(["Medication", "Medication_Class"])["patient"]
    .nunique()
    .reset_index()
    .rename(columns={"patient": "N_patients"})
    .sort_values("N_patients", ascending=False)
)

table_class.to_csv("Table_medication_classes.csv", index=False)
table_drug.to_csv("Table_medications_individual.csv", index=False)


comorbidity_dict = {
    "anxiety disorder": ["generalized anxiety disorder", "gad"],
    "ocd": ["ocd", "obsessive compulsive"],
    "ptsd": ["ptsd", "post traumatic stress disorder"],
    "bipolar disorder": ["bipolar"],
    "substance use disorder": ["substance", "alcohol misuse disorder", "drug abuse"],
    "adhd": ['attention deficit hyperactivity disorder', 'adhd'],
    'eating disorder': ['anorexia nervosa', 'bulimia nervosa']
}


def clean_participant_id(x):
    if pd.isna(x):
        return pd.NA
    match = re.search(r"(\d{3}R?)", str(x))
    return match.group(1) if match else pd.NA

def clean_text(x):
    if pd.isna(x):
        return ""
    x = x.lower()
    x = re.sub(r"[^a-z0-9\s]", " ", x)  # remove punctuation
    x = re.sub(r"\s+", " ", x)          # normalize spaces
    return x.strip()


def extract_medications(text):
    if pd.isna(text):
        return []

    text = text.lower()

    # split on separators between medications
    meds = re.split(r",|;|\+| and |\n", text)

    cleaned = []
    for med in meds:
        # remove dose + frequency info but NOT the medication name
        med = re.sub(r"\b\d+(\.\d+)?\s*(mg|ml|mcg|g)\b", "", med)
        med = re.sub(
            r"\b(once daily|twice daily|daily|bd|tds|nocte|mane|prn)\b",
            "",
            med
        )

        # remove leftover numbers
        med = re.sub(r"\b\d+\b", "", med)

        # collapse whitespace
        med = re.sub(r"\s+", " ", med).strip()

        if len(med) > 1:
            cleaned.append(med.title())

    return cleaned

