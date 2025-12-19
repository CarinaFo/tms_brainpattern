import os
import random
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import mne
from scipy.stats import spearmanr

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

BASE_PREPROC = r"L:\Lab_LucaC\Carina\tms_mdd\preprocessed_automatic_patientsupto209"
BASE_SOURCE  = r"L:\Lab_LucaC\Carina\TMS_MDD_project\source_reco_giles_automated"

patient_sessions = {}

for sess_dir in glob(os.path.join(BASE_PREPROC, "*_*")):
    sess_id = os.path.basename(sess_dir)   # e.g. "184_6"

    if not os.path.isdir(sess_dir):
        continue

    patient_id = sess_id.split("_")[0]

    # ICA file (inside session folder)
    ica_files = glob(os.path.join(sess_dir, "*_ica.fif"))
    if not ica_files:
        continue

    # RAW file (flat, NOT inside session folder)
    raw_crop_path = os.path.join(
        BASE_PREPROC, f"{sess_id}_preproc_crop-raw.fif"
    )
    if not os.path.exists(raw_crop_path):
        continue

    # Quick EOG check
    try:
        raw = mne.io.read_raw_fif(raw_crop_path, preload=False, verbose=False)
    except:
        continue

    if len(mne.pick_types(raw.info, eog=True)) == 0:
        continue

    # Source parcel data
    src_path = os.path.join(
        BASE_SOURCE, sess_id, "parc", "lcmv-parc-raw.fif"
    )
    if not os.path.exists(src_path):
        continue

    patient_sessions.setdefault(patient_id, []).append({
        "session": sess_id,
        "raw_path": raw_crop_path,
        "ica_path": ica_files[0],
        "src_path": src_path
    })


random.seed(42)

valid_patients = [p for p, s in patient_sessions.items() if len(s) > 0]
selected_patients = random.sample(valid_patients, 20)

print("Selected patients:")
for p in selected_patients:
    print(f"  Patient {p} ({len(patient_sessions[p])} sessions)")

all_results = []

for patient_id in selected_patients[:3]:
    print(f"\n=== Patient {patient_id} ===")

    for sess in patient_sessions[patient_id]:
        print(f"  Processing session {sess['session']}")

        raw = mne.io.read_raw_fif(sess["raw_path"], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(sess["ica_path"])
        raw_src = mne.io.read_raw_fif(sess["src_path"], preload=True, verbose=False)

        src_data = raw_src.get_data()

        # Find EOG ICs
        eog_inds, _ = ica.find_bads_eog(raw)
        if len(eog_inds) == 0:
            print("    ⚠ No EOG ICs found — skipping")
            continue

        # Eye IC time series
        eye_ts = (
            ica.get_sources(raw)
               .get_data(picks=eog_inds)
               .mean(axis=0)
        )

        # timepoints must match
        assert src_data.shape[1] == raw.get_data().shape[1]

        # Correlate frontal parcels
        for idx_1b, label in frontal_parcels.items():
            idx = idx_1b - 1
            r, p = spearmanr(src_data[idx, :], eye_ts)

            all_results.append({
                "patient": patient_id,
                "session": sess["session"],
                "parcel_index": idx_1b,
                "parcel_name": label,
                "spearman_r": r,
                "p_value": p,
                "n_eog_ic": len(eog_inds)
            })

df = pd.DataFrame(all_results)

parcels = sorted(df["parcel_name"].unique())

data = [
    df.loc[df["parcel_name"] == p, "spearman_r"].values
    for p in parcels
]

plt.figure(figsize=(10, 5))
plt.boxplot(data, labels=parcels, showfliers=True)
plt.axhline(0, linestyle="--")
plt.ylabel("Spearman r (parcel ↔ eye IC)")
plt.title("Distribution of eye–parcel correlations per parcel")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()
