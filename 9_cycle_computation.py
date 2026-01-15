"""Computes cyclical dynamics on HMM.
Fits tinda algorithm to state time course obtained from HMM (FO Asymmetry Matrix).
Runs permutation test on state time course and computes cycle strength.

Author: Carina Forster

Last update: 15/01/2026

Important: run in osld environment on linux
"""
import pickle
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import time
from joblib import Parallel, delayed
import os

# osl functions (clean that up)
from osl_dynamics.inference import modes
from osl_dynamics.analysis.tinda import tinda, optimise_sequence, circle_angles, compute_cycle_strength, plot_cycle

import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning) # tinda gives annoying warnings

# run in osld environment on neurov02 (requires tensorflow, currently only on linux setup)
print(f'we have {os.cpu_count()} CPUs')

# restrict to one thread
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# set working directory
base_dir = Path('/home/carinaf/LabData')

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

output_dir = Path(f"{hmm_dir}/figures/cycles")
output_dir.mkdir(parents=True, exist_ok=True)

# set up states and sessions
nses=6
nsubs=70

def calc_cycle_strength(cycle_allsessions: bool, ses_idx: int, n_states: int, 
                        permute_states: bool, n_permutations: int = 1000):
    """
    Computes the best sequence of HMM states for the group, each subject, and each session (code from Mats)

    Args:
        cycle_allsessions: boolean, do we want to compute the cycle for all sessions concatenated
        ses_idx: integer, which session do we want to analyse

    Returns:
        numpy array with cycle strengths
    """
    if cycle_allsessions:
        # we load the stc for each session (stacked in a list of patients, list of sessions, timeseries X states)
        # we use SUBJECT MAJOR ordering, e.g. patient 1 session 1, patient 1 session 2 etc.
        stc = concatenate_sessions_per_patient(nses, True, n_states)
        ses_idx = 99
    else:
        stc = pickle.load(open(Path(f"{hmm_dir}/states_{ses_idx}_{n_states}.pkl"), 'rb'))

    assert len(stc) == nsubs

    if cycle_allsessions:
        # flatten the list ( we have now subject X sessions stacked vertically, e.g. patient 1 session 1, patient 1 session 2...)
        stc_filtered = [session for patient in stc for session in patient]

        pickle.dump({"stc": stc_filtered}, open(f'{output_dir}/stc_{ses_idx}_{n_states}.pkl', 'wb'))

    fo = modes.fractional_occupancies(stc_filtered)  # shape (n_states, n_features)
    sns.boxplot(fo)
    plt.xlabel('States')
    plt.ylabel('Fractional Occupancy')
    plt.savefig(f'{output_dir}/fo_{ses_idx}_{n_states}.png')
    plt.show()

    # hard classify the state probabilities (state on or off, necessary for TINDA)
    stc_onoff = modes.argmax_time_courses(stc_filtered)

    # permute state time course state labels within each patient
    # and calculate cycle strength for each permutation
    if permute_states:

        start_time = time.time()

        n_jobs=20

        # run permutations in parallel
        null_model = Parallel(n_jobs=n_jobs, prefer='processes', verbose=10)(
            delayed(permute_state_time_course)(stc_onoff, n_states) for _ in range(n_permutations)
        )

        end_time = time.time()
        duration = end_time - start_time
        print(f"Permutation test for {str(n_permutations)} took ({duration/60:.2f} minutes)")

        # save the null model
        np.save(f'{output_dir}/null_model_{ses_idx}_{n_states}.npy', null_model)

        return null_model

    # apply tinda to stc
    fo_density, fo_sum, stats = tinda(stc_onoff)
    
    # now compute best sequence
    best_sequence = optimise_sequence(fo_density)

    # FO asymmetry (takes fo density and calculates difference between first and second interval and subtracts mean (normalizes the data))
    asym = np.squeeze((fo_density[:, :, 0] - fo_density[:, :, 1])/np.mean(fo_density, axis=2)) # shape is n_states, n_states, n_patients*n_sessions

    # should we plot this instead? normalizes FO asymmetry over patients
    mean_direction = np.squeeze(np.mean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=-1))

    mean_direction[np.isnan(mean_direction)] = 0

    # Prepare t-test outputs
    t_vals = np.zeros((n_states, n_states))
    p_vals = np.ones((n_states, n_states))

    # List for Bonferroni count
    tests = []

    # Run t-tests
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue

            vals = asym[i, j, :]

            t, p = ttest_1samp(vals, 0, nan_policy="omit")
            t_vals[i, j] = t
            p_vals[i, j] = p
            tests.append((i, j, p))

    # Bonferroni correction
    n_tests = len(tests)
    alpha_corr = 0.05 / n_tests
    significant = p_vals < alpha_corr

    # Plot mean asym (for this session only!)
    mean_asym = asym.mean(axis=-1)

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(mean_asym, cmap="coolwarm", center=0)

    # Add stars for significant edges
    for i in range(n_states):
        for j in range(n_states):
            if significant[i, j]:
                ax.text(j + 0.5, i + 0.5, "*",
                        ha="center", va="center",
                        color="black", fontsize=14, fontweight="bold")

    plt.title(f"FO Asymmetry – Significant Connections in session {ses_idx}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fo_asymmetry_{ses_idx}_{n_states}.png")
    plt.show()

    # now calculcate cycle strength (on a group basis)
    angleplot = circle_angles(best_sequence)

    # calculate cycle strength
    cyc_strength = compute_cycle_strength(angleplot, asym, 
                                                relative=True)
    # save observed cycle strength
    np.save(f'{output_dir}/observed_cycle_strength_{ses_idx}_{n_states}.npy', cyc_strength)

    pickle.dump({"fo_density":fo_density, "stats": stats, "best_sequence": best_sequence,
                    "asym": asym, "cycle_strength": cyc_strength, "mean_direction": mean_direction}, 
                open(f'{output_dir}/tinda_{ses_idx}_{n_states}.pkl', 'wb'))

    # plot the actual cycle
    plot_cycle(best_sequence, fo_density,  significant, new_figure=True)
    plt.title(f'Cycle in Session {ses_idx}')
    plt.savefig(f'{output_dir}/cycle_{ses_idx}_{n_states}.png')

    plt.show()

    return cyc_strength


def test_sign_cycle(test_group: bool = True, null_model: list = None, 
                          observed: np.ndarray = None, n_states: int = 6):
    
    # load null model
    null_model = np.load(f'{output_dir}/null_model_{ses_idx}_{n_states}.npy')
    observed = np.load(f'{output_dir}/observed_cycle_strength_{ses_idx}_{n_states}.npy')

    if test_group:
        # take mean over sample
        group_null = np.mean(np.array(null_model), axis=1) # shape is n_permutations
        group_observed = np.mean(observed) # scalar

        p_value = np.mean(group_null > group_observed)
        print(p_value)
    else:     
        # compare within each sample
        p_value = np.mean(np.array(null_model) > observed)
        print(p_value)

    # plot permutation test outcome
    plt.hist(group_null)
    plt.vlines(group_observed, 0, 250, color='red', label='observed')
    plt.legend()
    plt.savefig(f'{output_dir}/permutating the state time course 1000 times per subject for {n_states}')
    plt.show()

    return p_value


# Helper functions (move to utils at some point)

def concatenate_sessions_per_patient(nses, threed_array: bool = None, n_states=None):
    """ concatenate state time courses for all sessions and patients"""
    all_sessions=[]
    for session_idx in range(nses):
        # load state probabilities for each session
        state_probs = pickle.load(open(Path(f"{hmm_dir}/states_{session_idx}_{n_states}.pkl"), 'rb'))
        all_sessions.append(state_probs)

    per_patient_session = list(zip(*all_sessions))

    if threed_array:
        return [[np.array(session) for session in sessions] for sessions in per_patient_session]
    else:
        return [np.concatenate(sessions, axis=0) for sessions in per_patient_session]


def load_and_trim_stcs(sess_list: list = range(nses)):
    """ Load and trim all stcs to same length by removing from beginning and end."""
    all_sessions = []

    for session_idx in sess_list:
        # List of 40 patients: each entry is array [T_i, S]
        session_data = pickle.load(open(Path(f"{save_dir}/states_{session_idx}_{n_states}.pkl"), 'rb'))
        all_sessions.append(session_data)

    # Reorganize: list[patients][sessions] = array [T_i, S]
    per_patient_sessions = list(zip(*all_sessions))  # shape: [n_patients][n_sessions]

    # Find min T across all sessions for all patients
    min_len = min(
        session.shape[0]
        for patient in per_patient_sessions
        for session in patient
    )

    print(f"Minimum time length across all sessions: {min_len}")

    # Now trim all sessions to min_len by cutting from both ends
    trimmed = []
    for patient_sessions in per_patient_sessions:
        trimmed_patient = []
        for session in patient_sessions:
            T = session.shape[0]
            cut = (T - min_len) // 2
            trimmed_session = session[cut:cut + min_len, :]
            trimmed_patient.append(trimmed_session)
        trimmed.append(trimmed_patient)

    return trimmed  # shape: [n_patients][n_sessions] = array[T_min, S]


def order_states_based_on_coherence():

    for session_idx in range(n_sessions):

        # load frequencies
        f = np.load(f"{tp_dir}/f_0_{n_states}.npy")

        # load state probabilities for each session
        state_probs = pickle.load(open(Path(f"{tp_dir}/states_{session_idx}_{n_states}.pkl"), 'rb'))

        # load coherence for each session (reorder states based on coherence)
        coh = np.load(Path(f"{tp_dir}/coh_{session_idx}_{n_states}.npy"))  # (n_subjects, n_states, n_parcels, n_parcels, n_freq)

        # load NNMF weights
        wb_comp = np.load(Path(f"{tp_dir}/nnmf_{session_idx}_{n_states}.npy"))

        # load participant weights
        w = np.load(Path(f"{tp_dir}/w_{session_idx}_{n_states}.npy"))  # (n_subjects,)

        # calculate coherence averaged over participants
        gcoh = np.average(coh, axis=0, weights=w)

        # Calculate the coherence network by averaging over a frequency range
        # we apply NNMF to the data (2 factors)
        c = connectivity.mean_coherence_from_spectra(f, gcoh, wb_comp)

        # shape of connectivity matrix: n_compos, n_states, n_parcels, n_parcels
        cyc_strength_perms = run_TINDA(state_probs, None, True, 0, 100, 8)

          # compute mean coherence for each state TODO: show Cameron or Mats or Chet
        state_coherence_scores = compute_mean_coherence(coh, n_states)

        # sort by coherence with higher coherence = earlier state
        state_order = np.argsort(state_coherence_scores)[::-1]  # descending

        # reorder state time courses to compare to Cam's cycles (van Es et al., 2025)
        stc = [stc[:, state_order] for stc in stc]

        # save state mapping (old state is now new state x)
        state_mapping_df = pd.DataFrame({
                "new_state": np.arange(len(state_order)),
                "original_state": state_order
            })
        
        # Save to CSV
        state_mapping_df.to_csv(f"{tp_dir}/state_mapping_{session_idx}.csv", index=False)


def compute_mean_coherence(coh: np.ndarray = None, n_states: int = None):
    # Compute mean coherence (upper triangle only) for each state
    state_coherence_scores = []
    for s in range(n_states):
        upper = np.triu(coh[0, s, :, :], k=1) # we look at coherence for the first component
        mean_coh = upper[upper > 0].mean()
        state_coherence_scores.append(mean_coh)

    return state_coherence_scores


def permute_state_time_course(state_time_course: np.ndarray, n_states: int):

    # Permute
    permuted_state_time_course = [permute_state_sequences(stc, n_states) for stc in state_time_course]

    # TINDA
    fo_density, _, _ = tinda(permuted_state_time_course)

    # Best sequence
    best_sequence = optimise_sequence(fo_density)

    # Angles
    angleplot = circle_angles(best_sequence)

    # Asymmetry
    asym = np.squeeze(np.nanmean((fo_density[:, :, 0] - fo_density[:, :, 1]), axis=2))

    # Cycle strength
    cyc_strength = compute_cycle_strength(angleplot, asym, relative=False)

    return cyc_strength


def permute_state_sequences(state_sequences: np.ndarray, n_states: int = None):
    """
    Permute state labels by shuffling columns of a one-hot encoded state sequence.
    
    Parameters:
    - state_sequences (np.ndarray): array of shape (timepoints, n_states), one-hot encoded.
    - n_states (int): number of unique states (should match state_sequences.shape[1]).
    
    Returns:
    - permuted_sequences (np.ndarray): permuted state sequences.
    - permutations (np.ndarray): the permutation applied to the columns.
    """
    perm = np.random.permutation(n_states)  # new label for each original state
    permuted_seq = state_sequences[:, perm]  # apply column permutation

    return permuted_seq