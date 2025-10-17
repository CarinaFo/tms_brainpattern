
"""Script for preprocessing resting-state EEG data for the TMS_MDD_QNC study
    This script preprocesses the data in parallel.
    Run python file in terminal (after activating conda environment osl) for parallel processing to run smoothly.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Carina Forster <carina.forster@qimrb.edu.au>

# Code has been adapted but is based on the Lemon preprocessing pipeline implemented in osl ephys
# code written by Andrew Qinn and Chetan Gohil
# run in osl-e environment on linux (!!!!! does not run on windows)
import pandas as pd
import numpy as np
import mne

from dask.distributed import Client
import re
import os

from pathlib import Path

from osl_ephys import preprocessing, utils

# determine cores available 
print(f"Number of CPU cores: {os.cpu_count()}")
cpus_avail = os.cpu_count()

workstation = 'lucky3'

home_dir = 'home/carinaf/LabData/Lab_LucaC/Carina/tms_mdd'

# create output directory
outdir = f'{home_dir}/preprocessed_automatic'
os.makedirs(outdir, exist_ok=True)

# We will first fetch all data using an osl-ephys utility
file_name = '{subj}_{session}-raw'
fullpath = os.path.join(home_dir, 'raw_fif_files', file_name + '.fif')
datafiles = utils.Study(fullpath)
filenames = datafiles.get() # this should be the path not name?

# Get sorted list of file names
sorted_files = sorted(filenames)

# Extract participant IDs using regex
idlist = []
ids_only = [] # store ids only to get unique patients
# Define the pattern to capture 'PD_' followed by three digits (with optional 'R') and a session number
pattern = r"(\d{3}R?)_(\d)"  # Captures 'PD_123_1' or 'PD_123R_2'

for file in sorted_files:
    # Remove "(Withdrawn)" before processing
    cleaned_file = re.sub(r"\s*\(Withdrawn\)", "", file)  
    match = re.search(pattern, cleaned_file)
    if match:
        id = match.group(1)  # Store only the numeric ID
        ids_only.append(id)
        session = match.group(2) # store session
        idlist.append(f"{id}_{session}")

print(f'we have data for {len(pd.unique(ids_only))} patients')

# make sure IDlist is the same length as files 
assert len(idlist) == len(sorted_files)

if __name__ == "__main__":

    utils.logger.set_up(level="INFO")

    def create_heog(dataset, userargs):
    
        reye = dataset["raw"].get_data(picks="1RB")
        leye = dataset["raw"].get_data(picks="1LC")
    
        heog = reye - leye
    
        info = mne.create_info(["HEOG"], dataset["raw"].info["sfreq"], ["eog"])
        eog_raw = mne.io.RawArray(heog, info)
        dataset["raw"].add_channels([eog_raw], force_update_info=True)
    
        return dataset


    def custom_ica(dataset, userargs, logfile=None):
            ica = mne.preprocessing.ICA(
                n_components=userargs["n_components"], max_iter=1000, random_state=42
            )
            fraw = dataset["raw"].copy().filter(l_freq=1.0, h_freq=40)
            ica.fit(fraw, picks=userargs["picks"])
            dataset["ica"] = ica
            # Find and exclude VEOG
            veog_indices, eog_scores = dataset["ica"].find_bads_eog(dataset["raw"], ["1L", "1R"])
            dataset["veog_scores"] = eog_scores
            dataset["ica"].exclude.extend(veog_indices)
            # Find and exclude HEOG
            heog_indices, eog_scores = dataset["ica"].find_bads_eog(dataset["raw"], "HEOG")
            dataset["heog_scores"] = eog_scores
            dataset["ica"].exclude.extend(heog_indices)
             # Save components as channels in raw object
            src = dataset["ica"].get_sources(fraw).get_data()
            if heog_indices and veog_indices:
                heog = src[heog_indices[0], :]
                veog = src[veog_indices[0], :]
                ica.labels_["top"] = [veog_indices[0], heog_indices[0]]
                info = mne.create_info(
                ["ICA-VEOG", "ICA-HEOG"], dataset["raw"].info["sfreq"], ["misc", "misc"]
                )
                eog_raw = mne.io.RawArray(np.c_[veog, heog].T, info)
                dataset["raw"].add_channels([eog_raw], force_update_info=True)
            elif veog_indices and not heog_indices:
                veog = src[veog_indices[0], :]
                ica.labels_["top"] = [veog_indices[0]]
                info = mne.create_info(
                ["ICA-VEOG"], dataset["raw"].info["sfreq"], ["misc"]
                )
                eog_raw = mne.io.RawArray(np.c_[veog].T, info)
                dataset["raw"].add_channels([eog_raw], force_update_info=True)
            elif heog_indices and not veog_indices:
                heog = src[heog_indices[0], :]
                ica.labels_["top"] = [heog_indices[0]]
                info = mne.create_info(
                ["ICA-HEOG"], dataset["raw"].info["sfreq"], ["misc"]
                )
                eog_raw = mne.io.RawArray(np.c_[heog].T, info)
                dataset["raw"].add_channels([eog_raw], force_update_info=True)
            # Apply ICA denoising or not
            if ("apply" not in userargs) or (userargs["apply"] is True):
                dataset["ica"].apply(dataset["raw"])
            return dataset

    # configure settings pre manual correction
    config_text_pre = """
            preproc:
            - create_heog: None # create horizontal eye movements channel (extra function)
            - set_channel_types: {HEOG: eog, 1RC: eog, 1LC: eog} # mne wrapper
            - filter: {l_freq: 0.25, h_freq: 50, method: iir, iir_params: {order: 5, ftype: butter}} # mne wrapper
            - notch_filter: {freqs: 50 100} # mne wrapper
            - resample: {sfreq: 250} # mne wrapper
            - bad_segments: {segment_len: 500, picks: eeg, significance_level: 0.2}
            - bad_segments: {segment_len: 500, picks: eeg, mode: diff, significance_level: 0.2}
            - bad_channels: {picks: 'eeg', significance_level: 0.2} # osl wrapper uses generalized ESD test (osl wrapper)
            - bad_segments: {segment_len: 500, picks: eog, significance_level: 0.2} # generalized ESD test (osl wrapper)
            - custom_ica: {n_components: 0.99, picks: 'eeg'} # mne wrapper for fastica
            - interpolate_bads: {reset_bads: False} # keep information about bad channels in info # mne anonymous (runs mne function directly)
            - drop_channels: {ch_names: ['HEOG', 'ICA-VEOG', 'ICA-HEOG'], on_missing: 'ignore'} # mne anonymous
            - set_eeg_reference: {projection: true} # mne anonymous
            """

    # configure settings post manual correction
    config_text_post = """
            preproc:
            - create_heog: None # create horizontal eye movements channel (extra function)
            - set_channel_types: {HEOG: eog, 1RC: eog, 1LC: eog} # mne wrapper
            - filter: {l_freq: 0.25, h_freq: 50, method: iir, iir_params: {order: 5, ftype: butter}} # mne wrapper
            - custom_ica: {n_components: 0.99, picks: 'eeg'} # mne wrapper for fastica
            - interpolate_bads: {reset_bads: False} # keep information about bad channels in info # mne anonymous (runs mne function directly)
            - drop_channels: {ch_names: ['HEOG', 'ICA-VEOG', 'ICA-HEOG'], on_missing: 'ignore'} # mne anonymous
            - set_eeg_reference: {projection: true} # mne anonymous
            """
    
    client = Client(threads_per_worker=1, n_workers=int(cpus_avail/2))

    # process subjects with batch
    preprocessing.run_proc_batch(config_text_pre, sorted_files, idlist, outdir=outdir, overwrite=True, 
                                dask_client=True, extra_funcs=[custom_ica, create_heog])

    #run sequentially
    #for i in range(first_index, first_index+1):
    #    preprocessing.run_proc_chain(config_text_pre, sorted_files[i], idlist[i], outdir=outdir,
    #                     overwrite=True, extra_funcs=[custom_ica, create_heog])


# helper functions (store in utils at some point)
def copy_fif_files():
    import os
    import shutil

    patient_ids = extract_patient_ids()

    # Define source and destination directories
    src_dir = f"{home_dir}/preprocessed_harsher"
    dst_dir = f"{home_dir}/preprocessed_after_manual"

    # Make sure the destination directory exists
    os.makedirs(dst_dir, exist_ok=True)

    # Loop over each patient ID
    for id_ in patient_ids:
        src_file = os.path.join(src_dir, id_, f"{id_}_preproc_manual-raw.fif")
        dst_file = os.path.join(dst_dir, f"{id_}_preproc_manual-raw.fif")  # flat structure

        if os.path.exists(src_file):
            shutil.copy2(src_file, dst_file)
            print(f"Copied {src_file} → {dst_file}")
        else:
            print(f"⚠️ File not found: {src_file}")

    
def extract_patient_ids(src_dir: Path = Path("/home/carinaf/tms_mdd/preprocessed_harsher")):

    import os

    # List all patient IDs based on subdirectories containing a .fif file
    patient_ids = []

    # Loop through each subdirectory in src_dir
    for subfolder in os.listdir(src_dir):
        subfolder_path = os.path.join(src_dir, subfolder)
        if os.path.isdir(subfolder_path):
            # Look for the expected .fif file
            fif_file = os.path.join(subfolder_path, f"{subfolder}_preproc_manual-raw.fif")
            if os.path.isfile(fif_file):
                patient_ids.append(subfolder)

    return patient_ids
