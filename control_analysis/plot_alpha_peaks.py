# Save PSD for visual parcel
import os
import pandas as pd
from glob import glob
import mne
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

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

# Choose parcels after checking Giles38 labels
# https://osl-dynamics.readthedocs.io/en/latest/parcellations/giles38.html
visual_parcels = {
    1: "L V1",
    2: "R V1",
    3: "L visual",
    4: "R visual"
}

stim_parcels = {
    "inf_frontal_l": 22,
    "inf_frontal_r": 23,
    "acc_l": 34,
    "acc_r": 35,      
    "acc_midline": 37,
    "dorsomedial_pfc_l": 26,
    "dorsomedial_pfc_r": 27,
    "dlpfc_l": 28,
    "dlpfc_r": 29
}

source_path  = Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/source_reco_giles_parcel")

# load source data
patient_sessions = {}

for sess_dir in sorted(glob(os.path.join(source_path, "*_*")))[:-1]:

    sess_id = os.path.basename(sess_dir)

    patient_id = sess_id.split("_")[0]

    # Source parcel data
    src_path = os.path.join(
        source_path, sess_id, "parc", "lcmv-parc-raw.fif"
    )
    if not os.path.exists(src_path):
        continue

    patient_sessions.setdefault(patient_id, []).append({
        "session": sess_id,
        "src_path": src_path
    })

# load patient ID list
patient_ids = pd.read_csv(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"))

# ---- settings for 2 pages ----
n_patients = 70
per_page = 35
rows, cols = 7, 5  # 35 panels per page
fmin, fmax = 1, 20

# ---- parcels 1-4 ----
ch_names = ["parcel_1", "parcel_2", "parcel_3", "parcel_4"]

# ---- set up 2 figures ----
fig1, axes1 = plt.subplots(rows, cols, figsize=(7.2, 9.7), sharex=False, sharey=False)
fig2, axes2 = plt.subplots(rows, cols, figsize=(7.2, 9.7), sharex=False, sharey=False)
axes1 = axes1.flatten()
axes2 = axes2.flatten()

for idx, patient_id in enumerate(patient_ids['patient_id']):

    print(f"\n=== Patient {patient_id} ===")

    for sess in patient_sessions[patient_id]:
        if sess['session'][-1] == '1':
            raw_src = mne.io.read_raw_fif(sess["src_path"], preload=True, verbose=False)
            psd = raw_src.compute_psd(method="welch", fmin=fmin, fmax=fmax, picks="misc", verbose="ERROR")

            freqs = psd.freqs
            data = psd.get_data(picks="misc")  # shape: (n_channels, n_freqs)

            # ---- NEW: average parcels 1-4 in linear power ----
            ch_idxs = [psd.ch_names.index(n) for n in ch_names]
            curve = data[ch_idxs].mean(axis=0)

            # choose which page/axis
            if idx < per_page:
                ax = axes1[idx]
            else:
                ax = axes2[idx - per_page]

            ax.plot(freqs, curve, linewidth=0.8)
            ax.text(0.03, 0.95, f"P{idx+1:02d}", transform=ax.transAxes, fontsize=7, va="top")

            ax.set_xlim(fmin, fmax)

            # separate y-lims per plot (small padding)
            y0, y1 = curve.min(), curve.max()
            pad = 0.05 * (y1 - y0 + 1e-12)
            ax.set_ylim(y0 - pad, y1 + pad)

            raw_src.close()
            break  # stop after first session '1'
        else:
            continue

# clean up axes labels (only outer ones)
for fig, axes in [(fig1, axes1), (fig2, axes2)]:
    for i, ax in enumerate(axes):
        if i // cols < rows - 1:
            ax.set_xticklabels([])
        ax.tick_params(labelsize=8, length=3)

    # Better global labels
    fig.supxlabel("Frequency (Hz)", fontsize=12, y=0.04)
    fig.supylabel("PSD (a.u.)", fontsize=12, x=0.04)

    fig.subplots_adjust(hspace=0.3, wspace=0.45)

# save 2-page output (two separate PDFs)
fig1.savefig(Path("L:/Lab_LucaC/Carina\canonical_hmm_finalsample/figures/Supp_PSDs_page1_parcels1-4mean.png"), dpi=300, )
fig2.savefig(Path("L:/Lab_LucaC/Carina\canonical_hmm_finalsample/figures/Supp_PSDs_page2_parcels1-4mean.png"), dpi=300)

plt.show()
