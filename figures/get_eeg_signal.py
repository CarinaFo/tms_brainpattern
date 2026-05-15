import mne
import matplotlib.pyplot as plt

parcel_clean=mne.io.read_raw_fif(r"L:\Lab_LucaC\Carina\canonical_hmm_finalsample\source_reco_giles_parcel\094_2\parc\lcmv-parc-raw.fif")

acc = parcel_clean.get_data()[-1,:]
acc_pos = parcel_clean.get_data()[-2,:]
medpfc = parcel_clean.get_data()[-3,:]

plt.plot(medpfc[30000:30250])
plt.plot(acc[30000:30250])
plt.plot(acc_pos[30000:30250])

plt.savefig(r"L:\Lab_LucaC\Carina\canonical_hmm_finalsample\hmm_fits_05Hzcanonical_1Hzfiltered\figures\main_figures\eeg_signals.svg")
plt.show()
