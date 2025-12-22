import os
import random
import numpy as np
import pandas as pd
from glob import glob
import matplotlib.pyplot as plt
import mne
from scipy.stats import spearmanr
import pickle
from pathlib import Path

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

preproc_path = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/preprocessed")
source_path  = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/source_reco_giles")

patient_sessions = {}

for sess_dir in sorted(glob(os.path.join(preproc_path, "*_*"))):
    sess_id = os.path.basename(sess_dir)

    if not os.path.isdir(sess_dir):
        continue

    patient_id = sess_id.split("_")[0]

    # ICA file (inside session folder)
    ica_files = glob(os.path.join(sess_dir, "*_ica.fif"))
    if not ica_files:
        continue

    # RAW file (flat, NOT inside session folder)
    raw_crop_path = os.path.join(
        preproc_path, f"{sess_id}_preproc-raw.fif"
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
        source_path, sess_id, "parc", "lcmv-parc-raw.fif"
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
selected_patients = sorted(random.sample(valid_patients, 20))
np.save(Path('L:/Lab_LucaC/Carina/canonical_hmm_finalsample/eyemovement_test/test_patients.npy'), np.array(selected_patients))

print("Selected patients:")
for p in selected_patients:
    print(f"  Patient {p} ({len(patient_sessions[p])} sessions)")

# load stc
stc = pickle.load(open(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/eyemovement_test/states_0_6.pkl"), 'rb'))

all_results = []

for idx, patient_id in enumerate(selected_patients):
    print(f"\n=== Patient {patient_id} ===")

    if patient_id == '123':
        continue

    for sess in patient_sessions[patient_id]:
        print(f"  Processing session {sess['session']}")

        raw = mne.io.read_raw_fif(sess["raw_path"], preload=True, verbose=False)
        ica = mne.preprocessing.read_ica(sess["ica_path"])
        raw_src = mne.io.read_raw_fif(sess["src_path"], preload=True, verbose=False)
        stc_patient = stc[idx]
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
        assert stc_patient.shape[1] == raw.get_data().shape[1]

        r, p = spearmanr(stc_patient[:, 1], eye_ts)

        print(r, p)

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
