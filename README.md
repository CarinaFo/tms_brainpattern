# tms_brainpattern

Run the code in the dedicated python environment: osld.yml (Python 3.10)

Forster et al., 2026

Preprint: 
https://www.biorxiv.org/content/10.64898/2026.05.27.728312v1.abstract

Code to analyse 64-channel, resting state (eyes closed) EEG data and clinical data obtained from 70 participants that
underwent repetitive, personalised TMS treatment (20-30 sessions) to the left DLPF for treatment-resistant major depressive disorder at an outpatient clinic.

Each participant has 6 EEG recordings, collected during treatment at 3 separate timepoints (Session 1, Session 11, Session 20).

At each timepoint EEG resting state data (roughly 4 minutes) was collected immediately before and after a single TMS session.

EEG data was preprocessed and projected into source space using custom scripts that utilise MNE-Python and osl-ephys.

A canonical 10 state TDE-HMM (Gohil et al., 2026, HMM trained on CAM-CAN dataset) was then applied to the parcellated EEG data (39 parcels)
and a state time course was inferred for each participant and each session.

The pre-trained models can be found [here](https://github.com/OHBA-analysis/Canonical-HMM-Networks). Credits to Chetan Gohil.

Fractional occupancy, transition probabilites and cycle parameters were extracted from the 10 state HMM and related to baseline
symptom severity (sefl reported HADS-D) and acute TMS effects on those parameters were used to predict symptom improvement from Session 1 to
Session 11 and from Session 11 to Session 20.

Baseline regression matched the HADS-D score before the first EEG session, all other HADS-D scores were matched based on proximity (date) to the EEG session, preferably pre TMS. 

See the preprint for details.

Please contact Carina Forster regarding data sharing options:
carinaforster0611@gmail.com or carinaforster@qimrb.edu.au
