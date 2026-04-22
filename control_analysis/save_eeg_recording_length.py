from pathlib import Path
import numpy as np
import pandas as pd
import re

base_dir = r"L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_05Hz_1Hzfiltereddata"

def extract_eeg_recording_lengths(base_dir, sfreq=250):
    base_dir = Path(base_dir)

    patients_path = base_dir / "patients_fitted_for_this_hmm.csv"
    patients_df = pd.read_csv(patients_path)
    patients_df["patient_id"] = patients_df["patient_id"].astype(str)
    patient_ids = patients_df["patient_id"].tolist()

    rows = []

    def extract_index(path_obj):
        match = re.search(r"(\d+)", path_obj.stem)
        if match is None:
            raise ValueError(f"Could not extract numeric index from filename: {path_obj.name}")
        return int(match.group(1))

    for session_folder in sorted(base_dir.glob("badsegments_data_*")):
        session_num = int(session_folder.name.split("_")[-1])

        npy_files = sorted(
            session_folder.glob("*.npy"),
            key=extract_index
        )

        if len(npy_files) != len(patient_ids):
            raise ValueError(
                f"Mismatch in {session_folder.name}: "
                f"{len(npy_files)} arrays but {len(patient_ids)} patient IDs"
            )

        for i, npy_file in enumerate(npy_files):
            arr = np.load(npy_file, allow_pickle=True)
            patient = patient_ids[i]

            if arr.ndim == 1:
                n_samples = arr.shape[0]
            elif arr.ndim == 2:
                n_samples = max(arr.shape)
            else:
                n_samples = arr.shape[-1]

            rows.append({
                "patient": patient,
                "session": session_num,
                "n_samples": n_samples,
                "eeg_recording_length": n_samples / sfreq,
            })

    return pd.DataFrame(rows)

df = extract_eeg_recording_lengths(base_dir, sfreq=250)
df.to_csv(f'{base_dir}/eeg_recording_length.csv')