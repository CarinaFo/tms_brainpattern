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
import os
from pathlib import Path

import osl_ephys
import re
from dask.distributed import Client

# run in osle environment on neurov02 (requires tensorflow, currently only on linux setup)
print(f'we have {os.cpu_count()} CPUs')

# restrict to one thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# setup FSL
osl_ephys.source_recon.setup_fsl('/home/carinaf/fsl')

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
preproc_dir = Path(f'{local_dir}/preprocessed')

files = [str(f) for f in preproc_dir.rglob("*_preproc-raw.fif")]

# Strip everything after "_preproc"
ids_session = [
    Path(f).stem.replace("_preproc-raw", "")
    for f in files
]

# Extract subject ID (handles 016 vs 016R)
ids_only = pd.unique([
    s.split("_")[0]
    for s in ids_session
])

# Create a mapping of EEG sessions to MRI files
eeg_mri_mapping = {}

for session in ids_session:
    match = re.match(r'(\d{3})[R]?_\d+', session)
    if match:
        patient_id = match.group(1)
        eeg_mri_mapping[session] = mri_id_map.get(patient_id, 'standard')

prepro_ids = pd.unique([ids[:-2] for ids in eeg_mri_mapping.keys()])

print(f'we have {len(prepro_ids)} preprocessed patients')

# Generate the list of preprocessed file paths
preproc_files = sorted([
    os.path.join(preproc_dir, d, f"{d}_preproc-raw.fif")
    for d in os.listdir(preproc_dir)
    if os.path.isfile(os.path.join(preproc_dir, d, f"{d}_preproc-raw.fif"))
])

# get the ordered ID and session 
ordered_ids = [
    os.path.basename(os.path.dirname(f))
    for f in preproc_files
]

# order the dictionary based on the ID and session
ordered_dict = {
    id_: eeg_mri_mapping[id_]
    for id_ in ordered_ids
    if id_ in eeg_mri_mapping
}

# 5 patients have no individual MRI scan

# make sure the preprocessed files and the MRI's are aligned
assert list(ordered_dict.keys()) == ordered_ids
assert len(ordered_dict) == len(preproc_files)

missing_mri = set(ordered_ids) - set(eeg_mri_mapping)
missing_eeg = set(eeg_mri_mapping) - set(ordered_ids)

assert not missing_mri, f"Missing MRI for: {missing_mri}"
assert not missing_eeg, f"Missing EEG for: {missing_eeg}"

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
            reg: 0.05 # regularize for rank deficiencies
            rank: {eeg: 50}# rank should be higher than parcels
            parcellation_file: parcellations/fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii
            method: spatial_basis
            orthogonalisation: symmetric
    """

    client = Client(threads_per_worker=1, n_workers=10)

    osl_ephys.source_recon.run_src_batch(
          config,
          outdir = outdir,
          subjects= list(ordered_dict.keys()),
          smri_files = list(ordered_dict.values()),
          preproc_files = preproc_files,
          dask_client=True
    )

#for i in range(len(preproc_files)):
#         osl_ephys.source_recon.run_src_chain(
#            config,
#             outdir = outdir,
#           subject = list(ordered_dict.keys())[i],
#           smri_file = list(ordered_dict.values())[i],
#           preproc_file = preproc_files[i]
#          )
         
# osl_ephys.source_recon.run_src_chain(
#     config,
#     outdir = outdir,
#     subject = list(ordered_dict.keys())[0],
#     smri_file = list(ordered_dict.values())[0],
#     preproc_file = preproc_files[0]
# )