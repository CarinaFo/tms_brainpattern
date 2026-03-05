# tms_brainpattern

Forster et al., 2026

Code to analyse 64-channel, resting state (eyes closed) data and clinical data obtained from 70 participants that
underwent repetitive, personalised TMS treatment (20-30 sessions) to the left DLPF for treatment-resistant major depressive disorder.

Each participant has 6 EEG recordings, collected during treatment at 3 separate timepoints (baseline, mid treatment, post treatment).

At each timepoint EEG resting state data (roughly 4 minutes) was collected immediately before and after a TMS session.

EEG data was preprocessed and projected into source space using custom scripts that utilise MNE-Python and osl-ephys.

Clean, source reconstructed EEG data was then applied to a canonical 10 state Hidden Markov Model.

Fractional occupancy, transition probabilites and cycle parameters were extracted from the HMM and related to baseline
symptom severity (HADS-D) and acute TMS effects on those parameters were used to predict symptom improvement from baseline to
mid-treatment and from mid-treatment to end of treatment.

The manuscript is currently prepared for publication.
