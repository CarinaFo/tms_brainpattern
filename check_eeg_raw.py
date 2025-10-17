import mne
from pathlib import Path
import os
import matplotlib.pyplot as plt

# run in MNE environment

patient_id = 'D_212'
home_dir = "L:\Lab_LucaC"
folder_path = f"{home_dir}\A_QNC_ANT_Data\TMS_MDD_EEG_data\{patient_id}"

vhdr_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.vhdr')])
print(f"Found {len(vhdr_files)} files:")
print(vhdr_files)

raw_list = []

for f in vhdr_files[4:]:
    file_path = os.path.join(folder_path, f)
    print(f"Loading {file_path} ...")
    
    raw = mne.io.read_raw_brainvision(file_path, preload=True)
    raw_list.append(raw)

print("All files loaded!")

for raw in raw_list:
    raw.filter(1,40).plot(block=True)

    raw.filter(1,40).compute_psd(fmax=45, exclude='bads').plot()
    plt.show()

# Re-reference (depends on your lab setup)
raw.set_eeg_reference("average")

# Downsample (optional, speeds ICA)
raw.resample(250)

# load EEG montage
montage = setup_channel_montage()

# attach channel locations to raw data
raw.set_montage(montage)

from mne.preprocessing import ICA

# Create ICA object
ica = ICA(n_components=20, random_state=97, method="fastica")  # or "infomax"

# Fit ICA on continuous data
ica.fit(raw)

# Plot topographies of ICA components
ica.plot_components()

# Click on components in the figure to open their time-series and properties
ica.plot_properties(raw, picks=[0, 1, 2])  # inspect a few components


def setup_channel_montage():

    # read in the channel montage x,y,z file provided from ANT Neuro (in mm)
    montage = mne.channels.read_custom_montage(Path(rf'{home_dir}\Carina\tms_mdd\NA-261_NoRef.xyz'))
    
    # Get current montage positions
    pos = montage.get_positions()

    # Convert ch_pos to meters for MNE
    ch_pos_m = {ch: coord / 1000.0 for ch, coord in pos['ch_pos'].items()}

    # Rebuild montage with updated ch_pos
    montage_updated = mne.channels.make_dig_montage(
        ch_pos=ch_pos_m,
        coord_frame='unknown' # fallback to 'head' if unknown
    )

    montage_updated.plot()

    return montage_updated