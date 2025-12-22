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

# run in osld environment on neurov02 (requires tensorflow, currently only on linux setup)
print(f'we have {os.cpu_count()} CPUs')

# restrict to one thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# setup FSL
#osl_ephys.setup_fsl('/usr/local/fsl')

glasser_parcellation = 'Glasser52_binary_space-MNI152NLin6_res-8x8x8.nii'
giles_39 = 'fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii'


# set working directory (L drive is too slow)
home_dir = '/home/carinaf/canonical_hmm_finalsample'
# anatomical has to be stored locally
local_dir = '/home/carinaf/canonical_hmm_finalsample'

outdir = os.path.join(local_dir, "source_reco_giles_parcel")

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
preproc_dir = f"{local_dir}/preprocessed"

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


print(f'we exclude {len(exclude_not_all_sessions)} patients')

# Filter out excluded patients from the dictionary
filtered_dict = {
    id_: struct
    for id_, struct in eeg_mri_mapping.items()
    if id_.split('_')[0] in prepro_ids and id_.split('_')[0]
}

# Generate the list of preprocessed file paths
preproc_files = [f"{preproc_dir}/{id_}_preproc-raw.fif" for id_ in filtered_dict.keys()]

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

    client = Client(threads_per_worker=1, n_workers=10)

    osl_ephys.source_recon.run_src_batch(
          config,
          outdir = outdir,
          subjects= list(filtered_dict.keys()),
          smri_files = list(filtered_dict.values()),
          preproc_files = preproc_files,
          dask_client=True
    )

# process single patients (debugging) 
# import numpy as np

# # get's you the first session index of that patient
#index = np.where(np.char.find(preproc_files, '141_2') != -1)[0][0]

#for i in range(index, len(preproc_files)):
#         osl_ephys.source_recon.run_src_chain(
#            config,
#             outdir = outdir,
#            subject = list(filtered_dict.keys())[i],
#            smri_file = list(filtered_dict.values())[i],
#            preproc_file = preproc_files[i]
#          )