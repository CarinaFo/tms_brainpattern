"""fit data to canonical HMM. Save HMM features

Author: Carina Forster,
        Chet Gohil

run in osld (Python 3.12) environment on linux (neuroserv2)
"""
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import statsmodels.formula.api as smf
import mne

# osl dynamics specific
import osl_dynamics
from osl_dynamics.models.hmm import Config, Model
from osl_dynamics.data import Data
from osl_dynamics.analysis import spectral

# set working directory
os.chdir(Path('/home/carinaf/canonical_hmm_finalsample'))

basedir = os.getcwd()
source_dir = os.path.join(basedir, "source_reco_giles_parcel")
prep_dir = os.path.join(basedir, "prepared_data_giles_05Hz_1Hzfiltereddata")
save_dir = os.path.join(basedir, "hmm_fits_05Hzcanonical_1Hzfiltered")
os.makedirs(save_dir, exist_ok=True)
os.makedirs(prep_dir, exist_ok=True)

missing_clinical_data = ['087', '112', '159', '215', '217', '218', '220', '221', '222', '223'] # 10 patients

noisy_eeg = ['117', '143', '160', '169', '179']

corrupt_MRI = ['141', '210']

no_individual_MRI = ['053R', '127', '135', '152', '189']

excluded_subjects = sorted({
    *missing_clinical_data,
    *noisy_eeg,
    *corrupt_MRI,
    *no_individual_MRI,
})

def load_canonical_hmm(n_states: int, parcellation: str, sequence_length: int = 400, batch_size: int = 64):
    
    means = np.load(f"{basedir}/cam-can/canonical_models_1Hz/{parcellation}/{n_states:02d}_states/means.npy")
    covs = np.load(f"{basedir}/cam-can/canonical_models_1Hz/{parcellation}/{n_states:02d}_states/covs.npy")
    trans_prob = np.load(f"{basedir}/cam-can/canonical_models_1Hz/{parcellation}/{n_states:02d}_states/trans_prob.npy")
    initial_state_probs = np.load(f"{basedir}/cam-can/canonical_models_1Hz/{parcellation}/{n_states:02d}_states//initial_state_probs.npy")

    config = Config(
        n_states=n_states,
        n_channels=means.shape[-1],
        sequence_length=sequence_length,
        learn_means=False,
        learn_covariances=True,
        initial_means=means,
        initial_covariances=covs,
        initial_trans_prob=trans_prob,
        initial_state_probs=initial_state_probs,
        batch_size=batch_size,
        learning_rate=0.01,  # we won't train the model, this hyperparameter doesn't matter
        n_epochs=20,  # we won't train the model, this hyperparameter doesn't matter
    )

    return Model(config)


def save_prep_data(parcellation: str = '38ROI_Giles'):

    # get all sessions
    id_list = sorted(os.listdir(source_dir))[:-2]

    # load preprocessed source data that
    filenames = []
    ids_before_exlusion=[]
    ids_after_exclusion=[]

    # load parcellation for each patient
    for id in id_list:
        ids = id[:-2]
        ids_before_exlusion.append(ids)
        if ids in excluded_subjects:
            # the PSDs are extremely noisy and/or EEG sessions/clinical scores are missing
            continue
        ids_after_exclusion.append(ids)
        file_name = os.path.join(source_dir, Path(f"{id}/parc/lcmv-parc-raw.fif"))
        filenames.append(file_name)

    # Create DataFrame with IDs and placeholder for patient data
    df = pd.DataFrame({
        'patient_id': pd.unique(pd.Series(ids_after_exclusion))
    })

    print(f'{len(pd.unique(pd.Series(ids_after_exclusion)))} patients prepared')

    # save patient IDs for later combining with clinical data
    df.to_csv(f'{prep_dir}/patients_fitted_for_this_hmm.csv', index=False)

    # Check which files exist or are missing
    missing_files = [f for f in filenames if not Path(f).exists()]

    if not missing_files:
        print("All files exist ✅")
    else:
        print("Missing files:")
        for f in missing_files:
            print(f)

    # Group into 6 session lists
    session_1 = filenames[0::6]
    session_2 = filenames[1::6]
    session_3 = filenames[2::6]
    session_4 = filenames[3::6]
    session_5 = filenames[4::6]
    session_6 = filenames[5::6]

    # Check lengths before asserting
    session_lengths = [len(session_1), len(session_2), len(session_3), 
                    len(session_4), len(session_5), len(session_6)]

    if not all(length == session_lengths[0] for length in session_lengths):
        print("Session length mismatch!")
        print(f"Session lengths: {session_lengths}")

    assert len(session_1) == len(session_2) == len(session_3) == len(session_4) == len(session_5) == len(session_6)

    all_sessions = [session_1, session_2, session_3, session_4, session_5, session_6]

    for session_idx, session in enumerate(all_sessions):
        
        # Load data
        data = Data(session, picks="misc", sampling_frequency=250, reject_by_annotation="omit", n_jobs=8)

        # drop bad segments before TDE
        bad_segments_removed_data = data.prepare({'filter': {'low_freq': 1}, 
                                                'remove_bad_segments': {"significance_level": 0.3,
                                                "maximum_fraction": 0.4, 
                                                "use_raw": False}})

        # save bad segments removed data for PSD
        bs_path=os.path.join(prep_dir, f'badsegments_data_{session_idx}')
        bad_segments_removed_data.save(bs_path)

        # TDE
        pca_components = np.load(f"{basedir}/cam-can/canonical_models_1Hz/{parcellation}/pca_components.npy")
        template_cov = np.load(f"{basedir}/cam-can/canonical_models_1Hz/{parcellation}/template_cov.npy")

        tde_data = bad_segments_removed_data.prepare({
            "align_channel_signs": {"template_cov": template_cov, "n_embeddings": 15},
            "tde_pca": {"n_embeddings": 15, "pca_components": pca_components},
            "standardize": {},
        })

        # save prepared data for each session
        tde_path=os.path.join(prep_dir, f'tde_data_{session_idx}')
        tde_data.save(tde_path)

    return 'saved tde data for all sessions'


def save_state_probabilities(session_idx: int = 0, n_states: int = 10):

    # Load model
    model = load_canonical_hmm(n_states=n_states, parcellation='38ROI_Giles')

    # load prepared data
    tde_path = os.path.join(prep_dir, f'tde_data_{session_idx}')
    session_data = Data(tde_path, n_jobs=8)

    # extract state probabilites and save them
    stc = model.get_alpha(session_data)

    pickle.dump(stc, open(f"{save_dir}/states_{session_idx}_{n_states}.pkl", "wb"))
    
    # Calculate a state time course by taking the most likely state
    stc = osl_dynamics.inference.modes.argmax_time_courses(stc)

    # Calculate transition probability matrices
    tp = osl_dynamics.analysis.post_hoc.calc_trans_prob_matrix(stc, n_states=n_states)
    np.save(f"{save_dir}/tp_{session_idx}_{n_states}.npy", tp)

    return f"saved tp and stc for session {session_idx}"


def save_hmm_features(session_idx: int = 0, n_states: int = 6):

    stc = pickle.load(open(f"{save_dir}/states_{session_idx}_{n_states}.pkl", 'rb'))

    # Calculate a state time course by taking the most likely state
    stc = osl_dynamics.inference.modes.argmax_time_courses(stc)

    # Fractional occupancy
    fo = osl_dynamics.inference.modes.fractional_occupancies(stc)

    # Mean lifetime
    lt = osl_dynamics.inference.modes.mean_lifetimes(stc, sampling_frequency=250)

    # Mean interval
    intv = osl_dynamics.inference.modes.mean_intervals(stc, sampling_frequency=250)

    # Mean switching rate
    sr = osl_dynamics.inference.modes.switching_rates(stc, sampling_frequency=250)

    # Save
    np.save(f"{save_dir}/fo_{session_idx}_{n_states}.npy", fo)
    np.save(f"{save_dir}/lt_{session_idx}_{n_states}.npy", lt)
    np.save(f"{save_dir}/intv_{session_idx}_{n_states}.npy", intv)
    np.save(f"{save_dir}/sr_{session_idx}_{n_states}.npy", sr)

    return f"saved hmm summary stats for session {session_idx}"


def save_spectral(session_idx: int = 0, n_states: int = 6):

    # load bad segments removed data for PSD
    bs_path=os.path.join(prep_dir, f'badsegments_data_{session_idx}')
    session_data = Data(bs_path, n_jobs=8)

    # we need to trim the cleaned source data to match the train data
    trimmed_data = session_data.trim_time_series(n_embeddings=15,
                                                sequence_length=400,
                                                prepared=False)
    
    # load state time course
    stc = pickle.load(open(Path(f"{save_dir}/states_{session_idx}_{n_states}.pkl"), 'rb'))
    
    # state probs and trimmed data need to have the same shape for TFR
    for a, x in zip(stc, trimmed_data):
        assert(a.shape[0] == x.shape[0])

    # Calculate multitaper spectra for each state and subject 
    f, psd, coh, w = spectral.multitaper_spectra(
        data=trimmed_data,
        alpha=stc,
        sampling_frequency=250,
        time_half_bandwidth=4,
        n_tapers=7,
        frequency_range=[3, 40],
        return_weights=True,
        n_jobs=16,
        standardize=True
    )

    np.save(Path(f"{save_dir}/f_{session_idx}_{n_states}.npy"), f)
    np.save(Path(f"{save_dir}/psd_{session_idx}_{n_states}.npy"), psd)
    np.save(Path(f"{save_dir}/coh_{session_idx}_{n_states}.npy"), coh)
    np.save(Path(f"{save_dir}/w_{session_idx}_{n_states}.npy"), w)

    # We fit 2 'wideband' components (NNMF)
    wb_comp = spectral.decompose_spectra(coh, n_components=2)

    np.save(Path(f"{save_dir}/nnmf_{session_idx}_{n_states}.npy"), wb_comp)

    return f"saved PSDs for session {session_idx}"


def compare_free_energy(session_idx: int, n_states: int):

    # load prepared data
    tde_path = os.path.join(prep_dir, f'tde_data_{session_idx}')
    session_data = Data(tde_path, n_jobs=8)

    # Load canonical HMM
    model = load_canonical_hmm(n_states=n_states, parcellation='38ROI_Giles')
    
    patient_fe = []
    for sub_idx, subj in enumerate(session_data):
        fe = model.free_energy(subj)  # free energy per patient
        patient_fe.append(fe)  # normalize by time points
    
    return patient_fe


n_sessions = 6
states = [6, 8, 10]

session_free_energy_states = []

for st in states:
    session_free_energy = []
    for i in range(n_sessions):
        save_state_probabilities(i, st)
        save_hmm_features(i, st)
        save_spectral(i, st)
        patient_fe = compare_free_energy(i, st)
        session_free_energy.append(patient_fe)
    session_free_energy_states.append(session_free_energy)

fe_array = np.squeeze(np.array(session_free_energy_states))

# free energy should decrease with model complexity
means = np.mean(fe_array, axis=1)
plt.figure(figsize=(8, 5))
plt.boxplot([6, 8, 10], means)
plt.xlabel('Number of HMM States')
plt.ylabel('Mean Free Energy')
plt.title('Free Energy vs Model Complexity')
plt.grid(True, alpha=0.3)

plt.show()

# Print the values
for i, states in enumerate([6, 8, 10]):
    print(f"{states} states: {means[i]:.3f}")

investigate_free_energy(session_free_energy)


def investigate_free_energy(session_free_energy: np.ndarray):

    # load clinical data dataframe
    csv_path = Path(f"{prep_dir}/hmm_demo_quest_6.csv")
    df_clin = pd.read_csv(csv_path)
    # Get unique patient-session combinations (removes the 8 duplicate rows per state)
    df_clin_unique = df_clin.drop_duplicates(subset=['patient', 'tms', 'session'])[['patient', 'session', 'dep_hads', 'madrs_score',
                                            'responder', 'tms']]
    
    n_sessions, n_patients = np.array(session_free_energy).shape
    
    fe = np.array(session_free_energy)

    patient_ids = df_clin['patient'].unique()

    df = pd.DataFrame({
        "subject":   np.repeat(patient_ids, n_sessions),
        "session":   np.tile(np.arange(n_sessions), n_patients),
        "fe": fe.T.flatten(),
    })

    # Create mapping from FE session index to clinical session + tms
    session_mapping = {
        0: {'session': 1, 'tms': 'pre'},   # session 0 -> session 1 pre
        1: {'session': 1, 'tms': 'post'},  # session 1 -> session 1 post  
        2: {'session': 2, 'tms': 'pre'},   # session 2 -> session 2 pre
        3: {'session': 2, 'tms': 'post'},  # session 3 -> session 2 post
        4: {'session': 3, 'tms': 'pre'},   # session 4 -> session 3 pre
        5: {'session': 3, 'tms': 'post'},  # session 5 -> session 3 post
    }
    
    # Add clinical session and tms columns to df
    df['clin_session'] = df['session'].map(lambda x: session_mapping.get(x, {}).get('session'))
    df['tms'] = df['session'].map(lambda x: session_mapping.get(x, {}).get('tms'))
    
    # Merge with clinical data using session + tms
    df_merged = df.merge(df_clin_unique, 
                        left_on=['subject', 'clin_session', 'tms'], 
                        right_on=['patient', 'session', 'tms'], 
                        how='left')
    
    df = df_merged
    # exclude noisy patients and repeater 
    exclude_ids = ['144R', '127', '182']
    df = df[~df['subject'].isin(exclude_ids) & 
            ~df['subject'].str.contains('R', na=False)]

    # Calculate mean free energy per patient
    patient_fe_mean = df.groupby('subject')['fe'].mean().sort_values()

    print("Patients with LOWEST free energy:")
    print(patient_fe_mean.head(5))  # Bottom 5
    print(f"\nLowest: Patient {patient_fe_mean.index[0]} with FE = {patient_fe_mean.iloc[0]:.4f}")

    print("\nPatients with HIGHEST free energy:")
    print(patient_fe_mean.tail(5))  # Top 5  
    print(f"\nHighest: Patient {patient_fe_mean.index[-1]} with FE = {patient_fe_mean.iloc[-1]:.4f}")

    # Get the actual min/max patients
    lowest_patient = patient_fe_mean.idxmin()
    highest_patient = patient_fe_mean.idxmax()
    print(f"\nLowest FE patient: {lowest_patient}")
    print(f"Highest FE patient: {highest_patient}")

    # Add group info
    patient_groups = df.groupby('subject')['responder'].first()
    patient_summary = pd.DataFrame({
        'mean_fe': patient_fe_mean,
        'responder': patient_groups
    })

    print("Bottom 5 patients:")
    print(patient_summary.sort_values('mean_fe').head())
    print("\nTop 5 patients:")
    print(patient_summary.sort_values('mean_fe').tail())

    # Plot trajectories of min/max patients
    extreme_patients = [lowest_patient, highest_patient]
    df_extreme = df[df['subject'].isin(extreme_patients)]

    plt.figure(figsize=(8,5))
    sns.lineplot(data=df_extreme, x="session_x", y="fe", hue="subject", marker='o')
    plt.title("Trajectories of patients with extreme free energy")
    plt.show()
        
    # plot free energy over sessions
    plt.figure(figsize=(8,5))
    sns.lineplot(data=df, x="session_x", y="fe", hue="subject",
                estimator=None, alpha=0.3, linewidth=0.8, legend=False)  # spaghetti lines
    sns.lineplot(data=df, x="session_x", y="fe",
                color="black", linewidth=2, legend=False)       # group mean ± SEM
    plt.xlabel("Session")
    plt.ylabel("Free energy (normalised)")
    plt.title("Free energy per session")
    plt.tight_layout()
    plt.show()

    df_sess1 = df[(df.session_x == 0)]

    # does free energy differ between responder?
    model = smf.ols(
        "np.log(fe) ~  dep_hads",
        data=df_sess1,
        groups="subject",            # random intercept per subject
    )
    result = model.fit()
    print(result.summary())

    # plot difference in fe
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df, x="responder", y="fe")  # spaghetti lines
    plt.xlabel("Session")
    plt.ylabel("Free energy (normalised)")
    plt.title("Free energy per session")
    plt.tight_layout()
    plt.show()


def save_recording_info():

    # get all sessions
    id_list = sorted(os.listdir(source_dir))[:-2]

    durations = []   # seconds
    rows = []

    for id in id_list:
        ids = id[:-2]

        if ids in excluded_subjects:
            continue

        fif_path = Path(source_dir) / id / "parc" / "lcmv-parc-raw.fif"
        if not fif_path.exists():
            print(f"Missing: {fif_path}")
            continue

        # preload=False is faster + lower memory when you only need metadata
        raw = mne.io.read_raw_fif(fif_path, preload=False, verbose="ERROR")

        dur_sec = raw.n_times / raw.info["sfreq"]
        durations.append(dur_sec)

        rows.append({
            "patient": ids,
            "file": str(fif_path),
            "sfreq": raw.info["sfreq"],
            "n_times": raw.n_times,
            "duration_sec": dur_sec,
            "duration_min": dur_sec / 60.0,
        })

    # Per-file table (nice to save/check)
    df_dur = pd.DataFrame(rows)

    # Summary stats
    summary = {
        "n_files": len(durations),
        "mean_sec": float(np.mean(durations)) if durations else np.nan,
        "median_sec": float(np.median(durations)) if durations else np.nan,
        "min_sec": float(np.min(durations)) if durations else np.nan,
        "max_sec": float(np.max(durations)) if durations else np.nan,
    }
    summary["mean_min"] = summary["mean_sec"] / 60.0
    summary["median_min"] = summary["median_sec"] / 60.0
    summary["min_min"] = summary["min_sec"] / 60.0
    summary["max_min"] = summary["max_sec"] / 60.0

    print("Summary (seconds):", {k: summary[k] for k in ["n_files","mean_sec","median_sec","min_sec","max_sec"]})
    print("Summary (minutes):", {k: summary[k] for k in ["mean_min","median_min","min_min","max_min"]})

    # Optional: save outputs
    out_csv = Path(source_dir) / "recording_durations.csv"
    df_dur.to_csv(out_csv, index=False)
    print("Saved per-file durations:", out_csv)
