import pandas as pd
from pathlib import Path
import re

original_file = Path(r"L:\Lab_LucaC\A_QNC_Databank\Participants_Clinical_TMS_Data\21042026 MDD Anonymised QNC Clinical Data.xlsx")
edited_file = Path(r"L:\Lab_LucaC\Carina\canonical_hmm_finalsample\21042026 MDD Anonymised QNC Clinical Data_carinassample.xlsx")

def raw(x):
    if pd.isna(x):
        return ""
    return str(x).strip()


old = pd.read_excel(original_file, sheet_name="HADS")
new = pd.read_excel(edited_file, sheet_name="HADS")

id_old = old.columns[0]
id_new = new.columns[0]

old[id_old] = old[id_old].astype(str).str.strip()
new[id_new] = new[id_new].astype(str).str.strip()

old["Session"] = old["Session"].astype(str).str.strip()
new["Session"] = new["Session"].astype(str).str.strip()

# add duplicate row counter within participant + session
old["_dup"] = old.groupby([id_old, "Session"]).cumcount()
new["_dup"] = new.groupby([id_new, "Session"]).cumcount()

old = old.set_index([id_old, "Session", "_dup"])
new = new.set_index([id_new, "Session", "_dup"])

common_idx = old.index.intersection(new.index)
common_cols = old.columns.intersection(new.columns)

raw_changes = []

for idx in common_idx:
    for col in common_cols:
        if col == 'Computation Check':
            continue
        old_val = raw(old.loc[idx, col])
        new_val = raw(new.loc[idx, col])

        if old_val != new_val:
            raw_changes.append({
                "participant_id": idx[0],
                "session": idx[1],
                "duplicate_row": idx[2],
                "column": col,
                "old": old_val,
                "new": new_val,
            })

raw_changes = pd.DataFrame(raw_changes)

print(f"Found {len(raw_changes)} raw changes")
raw_changes


def check_hads_dates(
    excel_file,
    sheet_name="HADS",
    date_col="Date Completed",
    session_col="Session",
    min_date="2020-01-01",
    max_date="2030-12-31",
    output_csv=None,
):
    df = pd.read_excel(excel_file, sheet_name=sheet_name)

    id_col = df.columns[0]

    df[id_col] = df[id_col].astype(str).str.strip()
    df[session_col] = df[session_col].astype(str).str.strip()

    raw_dates = df[date_col]

    parsed_dates = pd.to_datetime(
        raw_dates,
        errors="coerce",
        dayfirst=True,
    )

    min_date = pd.Timestamp(min_date)
    max_date = pd.Timestamp(max_date)

    rows = []

    for i, (pid, session, raw, parsed) in enumerate(
        zip(df[id_col], df[session_col], raw_dates, parsed_dates)
    ):
        raw_str = "" if pd.isna(raw) else str(raw).strip()

        if raw_str == "":
            rows.append({
                "excel_row": i + 2,
                "participant_id": pid,
                "session": session,
                "issue": "missing_date",
                "raw_value": raw,
                "parsed_date": None,
            })
            continue

        if pd.isna(parsed):
            rows.append({
                "excel_row": i + 2,
                "participant_id": pid,
                "session": session,
                "issue": "unparseable_date",
                "raw_value": raw,
                "parsed_date": None,
            })
            continue

        # check text format only for values that were entered as strings
        if isinstance(raw, str):
            if not re.fullmatch(r"\d{1,2}/\d{1,2}/\d{4}", raw_str):
                rows.append({
                    "excel_row": i + 2,
                    "participant_id": pid,
                    "session": session,
                    "issue": "unexpected_date_format",
                    "raw_value": raw,
                    "parsed_date": parsed.date(),
                })

        if parsed < min_date or parsed > max_date:
            rows.append({
                "excel_row": i + 2,
                "participant_id": pid,
                "session": session,
                "issue": "date_out_of_expected_range",
                "raw_value": raw,
                "parsed_date": parsed.date(),
            })

    qc_df = df.copy()
    qc_df["_parsed_date"] = parsed_dates

    session_order = {
        "Pre": 0,
        "Week 1": 1,
        "Week 2": 2,
        "Week 3": 3,
        "Week 4": 4,
        "Post": 5,
    }

    qc_df["_session_order"] = qc_df[session_col].map(session_order)

    qc_df = qc_df.sort_values([id_col, "_session_order"])

    for pid, d in qc_df.groupby(id_col):
        d = d.dropna(subset=["_parsed_date", "_session_order"]).copy()

        if len(d) < 2:
            continue

        date_diff = d["_parsed_date"].diff().dt.days

        backwards = d.loc[date_diff < 0]

        for idx, row in backwards.iterrows():
            rows.append({
                "excel_row": idx + 2,
                "participant_id": row[id_col],
                "session": row[session_col],
                "issue": "date_decreases_relative_to_previous_session",
                "raw_value": row[date_col],
                "parsed_date": row["_parsed_date"].date(),
            })

    issues = pd.DataFrame(rows)

    if output_csv is not None:
        issues.to_csv(output_csv, index=False)

    return issues