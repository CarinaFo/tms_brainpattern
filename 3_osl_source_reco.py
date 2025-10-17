"""Coregisteration of individudal T1 image to EEG montage
using FSL and the osl-ephys package.
"""

# Authors: Chetan Gohil <chetan.gohil@psych.ox.ac.uk>
#          Carina Forster <carina.forster@qimrb.edu.au>

# run in osl-e on Linux (!! does not work in windows !!)
# get the latest developer version
#pip install git+https://github.com/OHBA-analysis/osl-ephys.git

# if you run on a headless server set this: export PYVISTA_OFF_SCREEN=true

import pandas as pd
import numpy as np
import os
import mne

import osl_ephys
import re
from dask.distributed import Client

# setup FSL
#osl_ephys.setup_fsl('/usr/local/fsl')

glasser_parcellation = 'Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii'
giles_39 = 'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii'

# determine cores available 
print(f"Number of CPU cores: {os.cpu_count()}")
cpus_avail = os.cpu_count()

# set working directory (L drive is too slow)
home_dir = '/home/carinaf/LabData/Lab_LucaC/Carina/tms_mdd'
# anatomical has to be stored locally
local_dir = '/home/carinaf/tms_mdd'

outdir = os.path.join(local_dir, "source_reco_giles_new_montage_automatic_upto209")

# change directory (important for parcellation, parcellations need to be stored there)
os.chdir(local_dir)

# structurals
smri_dir = os.path.join(local_dir, "anat")
os.makedirs(outdir, exist_ok=True)

# get all files in anat eeg directory
smri_files = sorted(os.listdir(smri_dir))

# Extract unique IDs from MRI filenames
mri_id_map = {}
for file in smri_files:
    match = re.search(r'D(\d{3})', file)
    if match:
        mri_id_map[match.group(1)] = f"{smri_dir}/{file}"

print(f'we have an MRI scan for {len(mri_id_map)} patients')

# Define the base directory for preprocessed files
preproc_dir = f"{local_dir}/preprocessed_automatic_patientsupto209"

# get all files that directory
filenames = sorted(os.listdir(preproc_dir))[:-2]

# Filter only the .fif files that contain '_crop_' in the filename
crop_fif_files = sorted([
    f for f in filenames
    if f.endswith('.fif') and 'crop' in f
])

# Strip everything after the second underscore to get id and session
ids_session = [key.split('_preproc')[0] for key in crop_fif_files]

# get an ID list
ids_only = pd.unique([i[:3] for i in ids_session])

# Create a mapping of EEG sessions to MRI files
eeg_mri_mapping = {}

for session in ids_session:
    match = re.match(r'(\d{3})[R]?_\d+', session)
    if match:
        patient_id = match.group(1)
        eeg_mri_mapping[session] = mri_id_map.get(patient_id, 'standard')


prepro_ids = pd.unique([ids[:-2] for ids in eeg_mri_mapping.keys()])

print(f'we have {len(prepro_ids)} preprocessed patients')

# based on manual inspection
# based on criteria matched with Ilya
exclude_patients = ['021R', '037R', '072', '067', '087',  '088', '090', '093', 
                    '094', '099', '101', '102', '106', '107', '108', '112',
                    '115', '113', '117', '118', '122', '123', '125', '127',
                    '135', '134', '137', '141', '145', '146R', '148', '152', '153',
                    '154', '156', '158', '159', '160', '161', '163',
                    '168', '171', '174', '178', '180', '184', '183', '189',
                    '190', '191', '194', '195', '198', '201']

exclude_not_all_sessions = ['090']

print(f'we exclude {len(exclude_not_all_sessions)} patients')

# Filter out excluded patients from the dictionary
filtered_dict = {
    id_: struct
    for id_, struct in eeg_mri_mapping.items()
    if id_.split('_')[0] in prepro_ids and id_.split('_')[0] not in exclude_not_all_sessions
}

# Generate the list of preprocessed file paths
preproc_files = [f"{preproc_dir}/{id_}_preproc_crop-raw.fif" for id_ in filtered_dict.keys()]

# lists need to be equal length (ID list, anatomical scan location, preprocessed fif files)
assert len(list(filtered_dict.keys())) == len(list(filtered_dict.values())) == len(preproc_files)

print(f'we have {int(len(filtered_dict)/6)} patients left')

if __name__ == "__main__":

    config = """
        source_recon:
        - extract_polhemus_from_info:
            include_eeg_as_headshape: true
        - compute_surfaces:
            include_nose: false
        - coregister:
            use_nose: false
            use_headshape: true
            n_init: 3
        - forward_model:
            model: Triple Layer
            eeg: true
            allow_smri_scaling: true
        - beamform_and_parcellate:
            freq_range: [1, 40]
            chantypes: eeg
            #reg: 0.05 # regularize for rank deficiencies
            rank: {eeg: 50}# rank should be higher than parcels
            parcellation_file: parcellations/fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii
            method: spatial_basis
            orthogonalisation: symmetric
    """

    # client = Client(threads_per_worker=1, n_workers=3)

    # osl_ephys.source_recon.run_src_batch(
    #      config,
    #      outdir = outdir,
    #      subjects= list(filtered_dict.keys()),
    #      smri_files = list(filtered_dict.values()),
    #      preproc_files = preproc_files,
    #      dask_client=True
    #  )

# process single patients (debugging) 
# import numpy as np

# # get's you the first session index of that patient
index = np.where(np.char.find(preproc_files, '141_2') != -1)[0][0]

for i in range(index, len(preproc_files)):
         osl_ephys.source_recon.run_src_chain(
            config,
             outdir = outdir,
            subject = list(filtered_dict.keys())[i],
            smri_file = list(filtered_dict.values())[i],
            preproc_file = preproc_files[i]
          )


def crop_data_for_hmm(fs: int = 250):

    import re

    # Define custom crop shifts (in seconds) for specific subjects (according to experimenter comements)
    special_crop_shifts = {
        "153_1": 25,
        "153_3": 60,
        "174_1": 40,
        "175_3": 15,
    }

    # get all files from the directory
    filenames = sorted(os.listdir(preproc_dir))[:-2]

    # Extract (id, session) tuples
    results = []
    pattern = re.compile(r"^([0-9]+R?|[0-9]{3})_([1-6])")

    for fname in filenames:
        match = pattern.match(fname)
        if match:
            subj_id = match.group(1)
            session = int(match.group(2))
            results.append((subj_id, session))

    # Load parcellation for each patient
    for subj_id, session in sorted(pd.unique(results)):
        
        # Skip subject IDs 180, 183, and all IDs >= 186 (different paradigm)
        if subj_id in ['180', '183'] or subj_id >= '186':

            id = f'{subj_id}_{str(session)}'

            file_name = os.path.join(preproc_dir, f"{id}/{id}_preproc-raw.fif")
        
            raw = mne.io.read_raw_fif(file_name, preload=True)

            annotations_pre_crop = raw.annotations

            # Crop first and last 1 minute
            cropped_raw = raw.copy().crop(tmin=60, tmax=raw.times[-1] - 60)

            annotations_post_crop = cropped_raw.annotations

            df_pre = pd.DataFrame({
                "onset_pre": annotations_pre_crop.onset,
                "duration_pre": annotations_pre_crop.duration,
                "description_pre": annotations_pre_crop.description,
            })
            
            df_post = pd.DataFrame({
                  "onset_post": annotations_post_crop.onset,
                "duration_post": annotations_post_crop.duration,
                "description_post": annotations_post_crop.description
            })

            annotations_path_pre = os.path.join(preproc_dir, f"{id}_annotations_pre.csv")

            df_pre.to_csv(annotations_path_pre, index=False)

            annotations_path_post = os.path.join(preproc_dir, f"{id}_annotations_post.csv")

            df_post.to_csv(annotations_path_post, index=False)

            file_name_crop = os.path.join(preproc_dir, f"{id}_preproc_crop-raw.fif")

            raw.save(file_name_crop, overwrite=True)

        else:

            id = f'{subj_id}_{str(session)}'

            file_name = os.path.join(preproc_dir, f"{id}/{id}_preproc-raw.fif")

            try:
                raw = mne.io.read_raw_fif(file_name, preload=True)

                if id in special_crop_shifts.keys():
                    # Determine shift
                    shift = special_crop_shifts.get(id, 0)
                else:
                    shift = 0
                
                if id == '087_1':
                    raw = raw
                else:
                    # Apply cropping
                    raw.crop(tmin=160, tmax=380 + shift)

                # save cropped file
                file_name_crop = os.path.join(preproc_dir, f"{id}_preproc_crop-raw.fif")
        
                # Save cropped data
                raw.save(file_name_crop, overwrite=True)

            except Exception as e:
                print(f"Skipping {id} due to error during cropping: {e}")


def update_montage(filename):
    """ANT Neuro finally provided a channel coordinates file"""

    # read in the channel montage x,y,z file provided from ANT Neuro (in mm)
    montage = mne.channels.read_custom_montage("/home/carinaf/LabData/Lab_LucaC/Carina/NA-261_NoRef.xyz")

    # Get current montage positions
    pos = montage.get_positions()

    # Convert ch_pos to meters for MNE
    ch_pos_m = {ch: coord / 1000.0 for ch, coord in pos['ch_pos'].items()}

    # Rebuild montage with updated ch_pos
    montage_updated = mne.channels.make_dig_montage(
        ch_pos=ch_pos_m,
        coord_frame='unknown' 
    )

    raw = mne.io.read_raw_fif(filename, preload=True)

    # interpolate bad channels
    raw.interpolate_bads(reset_bads=False)

    mapping = {'1LC': 'eeg', '1RC': 'eeg'}
    
    raw.set_channel_types(mapping)

    # set new montage
    raw.set_montage(montage_updated)

    # average reference
    raw.set_eeg_reference(projection=True)

    # Construct a new filename (e.g., add '_montage' before .fif)
    dirname, basename = os.path.split(filename)
    if "-raw.fif" in basename:
        new_basename = basename.replace("-raw.fif", "_new_montage-raw.fif")
    else:
        raise ValueError("Filename does not end with '-raw.fif': " + basename)

    new_filename = os.path.join(dirname, new_basename)

    # Save the updated file
    raw.save(new_filename, overwrite=True)