# group-level connectivity analysis in mne_connectivity

import mne
import mne_connectivity
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pickle


# --------------------------------------------------------
# setup directories
# --------------------------------------------------------
basedir = "/home/carinaf/tms_mdd"
source_dir = f"{basedir}/source_reco_glasser_new_montage"

# list of all participant folders
id_list = sorted(os.listdir(source_dir))[:-2]

# exclusions
exclude = ['092', '117', '123', '105', '160', '179']

# --------------------------------------------------------
# load Glasser52 parcel names
# --------------------------------------------------------
labels_path = f"{basedir}/parcellations/Labels.p"
parcel_names = pickle.load(open(labels_path, "rb"))
n_labels = len(parcel_names)

print("Loaded Glasser52 labels:", n_labels)


# ========================================================
# FUNCTION: compute connectivity for a single subject
# ========================================================
def compute_subject_connectivity(raw_file):
    """Load raw file, rename channels, compute PLI connectivity."""
    raw = mne.io.read_raw_fif(raw_file, preload=True)

    # sanity check — number of channels must match parcellation
    if len(raw.ch_names) != n_labels:
        print(f"Skipping {raw_file}: wrong number of channels ({len(raw.ch_names)})")
        return None

    # rename channels to Glasser52 parcels
    rename_dict = {old: new for old, new in zip(raw.ch_names, parcel_names)}
    raw.rename_channels(rename_dict)

    # band limits
    fmin, fmax = 8.0, 13.0
    sfreq = raw.info["sfreq"]

    # make epochs (required by connectivity estimator)
    events = mne.make_fixed_length_events(raw, duration=2.0)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=2.0,
                        baseline=None, preload=True, verbose=False)

    # compute PLI connectivity
    con = mne_connectivity.spectral_connectivity_epochs(
        epochs,
        method="pli",
        mode="multitaper",
        sfreq=sfreq,
        fmin=fmin,
        fmax=fmax,
        n_jobs=1,
    )

    # extract dense matrix [labels × labels]
    conmat = con.get_data(output="dense")[:, :, 0]  # freq=0
    return conmat


# ========================================================
# LOOP OVER ALL SUBJECTS
# ========================================================
all_conns = []

for subj in id_list:
    pid = subj[:-2]  # remove session suffix (if present)

    if pid in exclude:
        continue

    raw_path = f"{source_dir}/{subj}/parc/lcmv-parc-raw.fif"

    if not Path(raw_path).exists():
        print("Missing file:", raw_path)
        continue

    print("Processing:", raw_path)

    conmat = compute_subject_connectivity(raw_path)

    if conmat is not None:
        all_conns.append(conmat)


# ========================================================
# GROUP AVERAGE CONNECTIVITY
# ========================================================
all_conns = np.array(all_conns)

print("Shape of all_conns =", all_conns.shape)   # (n_subjects, 52, 52)

group_con = all_conns.mean(axis=0)
print("Group average connectivity computed.")


# ========================================================
# PLOT GROUP CONNECTIVITY
# ========================================================
mne_connectivity.viz.plot_connectivity_circle(
    group_con,
    parcel_names,
    n_lines=30,
    show=False,
)

plt.savefig('connectivity_alpha_band.png')
plt.show()

# optionally save
np.save(f"{basedir}/group_connectivity_glasser52.npy", group_con)
print("Saved group matrix to group_connectivity_glasser52.npy")
