# load EEG brainvision files and save as .fif files for further analysis
# make sure to load the channel layout template .xyz file with the corrdinates

# run in MNE_TMSMDD environment

import mne
from pathlib import Path
import numpy as np
import os

workstation = 'desktop'

if workstation == 'desktop':
    home_dir = 'L:/Lab_LucaC'
elif workstation == 'lucky3':
    home_dir = 'home/carinaf/LabData/Lab_LucaC'

# extract folder names from directory (participant IDs)
raw_data_path = Path(f'{home_dir}/A_QNC_ANT_Data/TMS_MDD_EEG_data')
out_dir = Path(f'{home_dir}/Carina/tms_mdd/raw_fif_files')

# Create the directory if it doesn't exist
os.makedirs(out_dir, exist_ok=True)

# remove the withdrawn from the file name
patient_ids_with_repeater = sorted([folder.name for folder in raw_data_path.iterdir() if folder.is_dir()])

print(f'We have {len(patient_ids_with_repeater)} patients so far (including responders that returned)')

# Create a dictionary to store .vhdr file counts
vhdr_counts = {}

for patient_id in patient_ids_with_repeater:
    patient_folder = raw_data_path / patient_id
    # Count .vhdr files (recursively if needed)
    vhdr_files = list(patient_folder.rglob("*.vhdr"))  # use rglob for recursive search
    vhdr_counts[patient_id] = len(vhdr_files)

# Optional: Print result
for pid, count in vhdr_counts.items():
    print(f"{pid}: {count} .vhdr files")

# Count how many patients have exactly 6 and less than 6 .vhdr files
full_sessions = [pid for pid, count in vhdr_counts.items() if count == 6]
partial_sessions = [pid for pid, count in vhdr_counts.items() if count < 6]
more_sessions = [pid for pid, count in vhdr_counts.items() if count > 6]

print(f"\nPatients with all 6 sessions: {len(full_sessions)}")
print(f"Patients with less than 6 sessions: {len(partial_sessions)}")
print(f"Patients with more than 6 sessions: {len(more_sessions)}")

assert len(full_sessions) + len(partial_sessions) + len(more_sessions) == len(patient_ids_with_repeater)


def save_raw_fif_file():

    """save raw fif file for each session (incl. channel montage)"""
    
    # load EEG montage
    montage = setup_channel_montage()

    # loop over patients
    for ids in np.sort(patient_ids_with_repeater):

        # patient D_113 has only 1 full session, the second sessions is useless (super noisy)
        # 54 has only 4 minute recording first session (no comment), different EEG cap session3?
        if ((ids == 'D_113') or (ids == 'D_054')):
            continue
        
        print(f'{ids} processed')

        # Order vhdr files by date and time in ascending order
        ids_folder = raw_data_path / ids  # Participant's folder path
        eeg_files = list(ids_folder.glob("*.vhdr"))  # Find all .vhdr files

        if (ids == "D_122") | (ids == "D_184"):
            fif_file = list(ids_folder.glob("*.fif"))  # Find .fif file (combined recording)
            eeg_files.extend(fif_file)
               # D122 has 3 recordings on 2023-11-10 (the first one is eyes open, 
               # then eyes close, open) then post session, I combined the first session into 1 file
               # patient D_184 has 2 recordings for the fourth session, combinded them

        # Sort the files based on extracted date and time
        eeg_files.sort(key=lambda f: extract_datetime_from_filename(f.name))
        
        # loop over EEG recordings
        for session, vhdr in enumerate(eeg_files):
            
            session += 1 # easier for non python users

            # where we save the raw fif file
            fif_out_path = Path(out_dir, f"P{ids}_{session}-raw.fif")

            if fif_out_path.exists():
                print(f"Skipping P{ids} session {session} — FIF already exists.")
                continue

            try:
                if ((ids == "D_184") & (session == 4)) | ((ids == "D_122") & (session == 3)):
                    # concateneded session files saved as .fif files
                    raw = mne.io.read_raw_fif(vhdr, preload=True)
                elif (((ids == 'D_090') & (session == 1)) or ((ids == 'D_123') & (session == 4))):
                    # only 4 minute recording, no comment in csv file
                    continue
                else:
                    raw = mne.io.read_raw_brainvision(vhdr, preload=True)

            except FileNotFoundError:
                # renamed file name but not header pointers
                fix_header_file(vhdr)
            
            # attach channel locations to raw data
            raw.set_montage(montage)

            # save raw filtered data
            raw.save(fif_out_path, overwrite=True)
            
    return "Done"


# Helper functions (will end up in utils.py at some point)
def fix_header_file(vhdr_path):
    
    import shutil

    # Extract the base filename (without extension)
    base_name = vhdr_path.stem  # Correctly gets filename without extension

    # Define the correct EEG and marker filenames
    correct_eeg_filename = f"{base_name}.eeg"
    correct_vmrk_filename = f"{base_name}.vmrk"

    # Create a backup of the original .vhdr file
    backup_vhdr_path = vhdr_path.with_name(f"{base_name}_renamed.vhdr")
    shutil.copy(vhdr_path, backup_vhdr_path)  # Make a backup

    print(f"Backup created: {backup_vhdr_path}")

    # Read the .vhdr file
    with vhdr_path.open("r") as file:
        lines = file.readlines()

    # Modify the DataFile and MarkerFile entries
    for i, line in enumerate(lines):
        if line.strip().startswith("DataFile="):
            old_filename = line.strip().split("=")[-1]
            lines[i] = f"DataFile={correct_eeg_filename}\n"
            print(f"Updated DataFile: {old_filename} → {correct_eeg_filename}")

        if line.strip().startswith("MarkerFile="):
            old_markerfile = line.strip().split("=")[-1]
            lines[i] = f"MarkerFile={correct_vmrk_filename}\n"
            print(f"Updated MarkerFile: {old_markerfile} → {correct_vmrk_filename}")

    # Overwrite the original .vhdr file with the fixed version
    with vhdr_path.open("w") as file:
        file.writelines(lines)

    print(f"Fixed header file saved: {vhdr_path}")


def setup_channel_montage():

    # read in the channel montage x,y,z file provided from ANT Neuro (in mm)
    montage = mne.channels.read_custom_montage(Path(f'{home_dir}/Carina/tms_mdd/NA-261_NoRef.xyz'))
    
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


# Function to extract date and time from filename
def extract_datetime_from_filename(filename):
    """Extracts date and time from a BrainVision .vhdr filename assuming format contains YYYY-MM-DD_HH-MM-SS."""
    import re
    from datetime import datetime
    
    match = re.search(r"(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})", filename)
    if match:
        date_str, time_str = match.groups()
        return datetime.strptime(f"{date_str} {time_str}", "%Y-%m-%d %H-%M-%S")  # Convert to datetime object
    return datetime.min  # Assign the oldest possible date if no match