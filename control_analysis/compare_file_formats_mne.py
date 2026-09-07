import mne 
from pathlib import Path

home_dir = 'L:/Lab_LucaC'

def setup_channel_montage():

    # read in the channel montage x,y,z file provided from ANT Neuro (in mm)
    montage = mne.channels.read_custom_montage(Path(f'{home_dir}/Carina/canonical_hmm_finalsample/channel_montage/NA-261_NoRef.xyz'))
    
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

# load EEG montage
montage = setup_channel_montage()
raw=mne.io.read_raw_brainvision("L:\Lab_LucaC\A_QNC_ANT_Data\TMS_MDD_EEG_data\D_251\D_251_2026-08-14_09-19-51.vhdr", preload=True)
raw_cnt= mne.io.read_raw_cnt(r"L:\Lab_LucaC\A_QNC_ANT_Data\TMS_MDD_EEG_data\D_251\wrong_montage\1_D_251_2026-07-27_09-19-35.cnt", preload=True)
raw_fif = mne.io.read_raw_fif("L:\Lab_LucaC\A_QNC_ANT_Data\TMS_MDD_EEG_data\D_251\D_251_2026-07-27_09-19-35.fif", preload=True)

raw.set_montage(montage)

raw.filter(1,40).plot()