# Run in MNE environment or base python
# check if frontal DMN state is mainly active during eye movements (ica component for eye movements)
import os
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

# windows paths
preproc_path = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/preprocessed")
source_path  = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/source_reco_giles_parcel")

patient_sessions = {}

for sess_dir in sorted(glob(os.path.join(preproc_path, "*_*")))[:-1]:

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
        preproc_path, f"{sess_id}/{sess_id}_preproc-raw.fif"
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

# load patient ID list
patient_ids = pd.read_csv(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"))

assert len(patient_ids) == 70

# load state time course for the first EEG session (baseline)
# state time course without bad segment rejection to match source and prepro data length
stc = pickle.load(open("L:\Lab_LucaC\Carina\canonical_hmm_finalsample\hmm_fits_05Hzcanonical_1Hzfiltered\states_0_10_nobadsegmentsrej.pkl", 'rb'))
# stc is missing 400 datapoints due to TDE

r_stc_eye_ic_corr, p_stc_eye_ic_corr  = [], []
all_results = []

# loop over patients analysed with HMM
for idx, patient_id in enumerate(patient_ids['patient_id']):

    print(f"\n=== Patient {patient_id} ===")

    for sess in patient_sessions[patient_id]:
        if sess['session'][-1] == '1':

            print(f"  Processing session {sess['session']}")

            raw = mne.io.read_raw_fif(sess["raw_path"], preload=True, verbose=False)
            ica = mne.preprocessing.read_ica(sess["ica_path"])
            raw_src = mne.io.read_raw_fif(sess["src_path"], preload=True, verbose=False)

            stc_patient = stc[idx]
            src_data = raw_src.get_data()

            # Find EOG ICs
            eog_inds, _ = ica.find_bads_eog(raw)
            
            # skip if no eogs
            if eog_inds == []:
                continue

            # Eye IC time series
            eye_ts = (
                ica.get_sources(raw)
                .get_data(picks=eog_inds)
                .mean(axis=0)
            )

            eyes_ts_trimmed = apply_same_trimming(np.array(eye_ts))

            assert eyes_ts_trimmed.shape[0] == stc_patient.shape[0]

            r, p = spearmanr(stc_patient[:, 1], eyes_ts_trimmed)

            r_stc_eye_ic_corr.append(r)
            p_stc_eye_ic_corr.append(p)
        else:
            continue

        assert src_data.shape[1] == eye_ts.shape[0]

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


# trim the data to match the state time course

def apply_same_trimming(array, sequence_length=400, n_embeddings=15):
    n_remove = n_embeddings // 2

    # 1. Remove embedding edges
    trimmed = array[n_remove:-n_remove]

    # 2. Remove remainder from sequencing
    n_keep = (trimmed.shape[0] // sequence_length) * sequence_length
    trimmed = trimmed[:n_keep]

    return trimmed
