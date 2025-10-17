# check clean files and manually remove bad channels
import os
import mne
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

prepro_dir = '/home/carinaf/tms_mdd/preprocessed_pre_manual'
after_manual_dir = '/home/carinaf/tms_mdd/preprocessed_after_manual'

# get all files in anat eeg directory
ids_list =sorted(os.listdir(prepro_dir))
id_only =[i[:6] for i in ids_list][:-2]

ids_cleaned = [id_[:-1] if id_.endswith('_') else id_ for id_ in id_only]

print(f'we have {len(pd.unique(id_only))} patients before exclusion')

# based on manual inspection of raw data and PSD
exclude_patients = ['021R', '037R', '072', '067', '087',  '088', '090', '093', 
                    '094', '099', '102', '106', '107', '108', '112',
                    '115', '113', '117', '118', '122', '123', '125',
                    '135', '134', '137', '141', '145', '146R', '148', '152',
                    '154', '156', '158', '159', '160', '161', '163',
                    '168', '171', '178', '180', '184', '183', '189', '190', '191',
                    '194', '195', '198', '201']

first_index = np.where(np.array(ids_cleaned)== '163_1')[0][0]


for i in ids_cleaned[first_index:first_index+6]:
   raw_path = os.path.join(prepro_dir, f'{i}/{i}_preproc-raw.fif')
   raw =  mne.io.read_raw_fif(raw_path, preload=True)
   raw.plot(n_channels=64, block=True)
   raw.compute_psd().plot()
   plt.show(block=True)
   #enter bad channels
   bads = input("Enter bad channels (comma-separated), or leave empty: ")
   if bads.strip():
      raw.info['bads'].extend([ch.strip() for ch in bads.split(',')])
   raw.interpolate_bads(reset_bads=False)
   raw.filter(1, 40)
   raw.set_eeg_reference(projection=True)
   after_manual_path = os.path.join(after_manual_dir, f'{i}_preproc_manual-raw.fif')
   raw.save(after_manual_path,  overwrite=True)
   fig = raw.compute_psd().plot(show=False)
   psd_path = os.path.join(after_manual_dir, f'psds/{i}_psd.png')
   plt.savefig(psd_path)


# ilyas_list = ['D_175', 'D_137', 'D_155', 'D_154','D_153', 'D_119', 'D_111','D_152',
#                'D_092', 'D_132', 'D_037R', 'D_165', 'D_130', 'D_163', 'D_162', 'D_016R',
#                'D_030R', 'D_140', 'D_110', 'D_101', 'D_105', 'D_108', 'D_143', 'D_130R',
#                'D_098', 'D_097', 'D_131', 'D_144', 'D_158', 'D_144R', 'D_150', 'D_127']

# sorted(ilyas_list)

# remaining_patients = [pat for pat in pd.unique(id_only) if pat not in exclude_patients]

# print(f'{len(remaining_patients)} made it into the next round')