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
base_dir = Path('L:/')

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

output_dir = Path(f"{hmm_dir}/figures/cycles")
output_dir.mkdir(parents=True, exist_ok=True)

nses=6 # we have 6 sessions
nsubs=70 # for 70 patients

def calc_cycle_strength(ses_idx: int, n_states: int, 
                        permute_states: bool, n_permutations: int, n_cores_avail: int):
    """
    Computes the best sequence of HMM states for the group, each subject, and each session (code from Mats)

    Args:
        cycle_allsessions: boolean, do we want to compute the cycle for all sessions concatenated
        ses_idx: integer, which session do we want to analyse
        n_states: integer, HMM brain states
        permute_states: bool, run permutation test, ATTENTION: adjust njobs depending on system
        Neuroserv2: 1000 permutations, 20 cores ~ 2hrs
        n_permutations: int, how many permutations(!!!!)
        n_cores_avail: int, how many cores do we have available?

    Returns:
        numpy array with cycle strengths
    """

    # we load the stc for each session (stacked in a list of patients, list of sessions, timeseries X states)
    # we use SUBJECT MAJOR ordering, e.g. patient 1 session 1, patient 1 session 2 etc.
    stc = concatenate_sessions_per_patient(nses, n_states, True)

    assert len(stc) == nsubs

    # flatten the list ( we have now subject X sessions stacked vertically, e.g. patient 1 session 1, patient 1 session 2...)
    stc_filtered = [session for patient in stc for session in patient]
    pickle.dump({"stc": stc_filtered}, open(f'{output_dir}/stc_{ses_idx}_{n_states}.pkl', 'wb'))

    fo = modes.fractional_occupancies(stc_filtered)  # shape (n_observations, n_states)
    sns.boxplot(fo)
    plt.xlabel('States')
    plt.ylabel('Fractional Occupancy')
    plt.savefig(f'{output_dir}/fo_{ses_idx}_{n_states}.png')
    plt.savefig(f'{output_dir}/fo_{ses_idx}_{n_states}.svg')
    plt.show()

    # hard classify the state probabilities (state on or off, necessary for TINDA)
    stc_onoff = modes.argmax_time_courses(stc_filtered)

    # permute state time course state labels within each patient
    # and save cycle strength for each permutation
    if permute_states:

        start_time = time.time()

        n_jobs=n_cores_avail

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
    fo_density, _, stats = tinda(stc_onoff)
    
    # now compute best sequence
    best_sequence = optimise_sequence(fo_density)

    # normalised FO asymmetry
    asym = compute_asym(fo_density)

    np.save(f'{output_dir}/fo_asymmetry_{ses_idx}_{n_states}.npy', asym)

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

    #plt.title(f"FO Asymmetry – Significant Connections in session {ses_idx}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fo_asymmetry_{ses_idx}_{n_states}.png")
    plt.savefig(f"{output_dir}/fo_asymmetry_{ses_idx}_{n_states}.svg")
    plt.show()

    # now calculcate cycle strength (on a group basis)
    angleplot = circle_angles(best_sequence)

    # calculate cycle strength (normalised)
    cyc_strength = compute_cycle_strength(angleplot, asym, relative=True)

    # save observed cycle strength
    np.save(f'{output_dir}/observed_cycle_strength_{ses_idx}_{n_states}.npy', cyc_strength)

    pickle.dump({"fo_density":fo_density, "stats": stats, "best_sequence": best_sequence,
                    "asym": asym, "cycle_strength": cyc_strength, "mean_direction": mean_direction}, 
                open(f'{output_dir}/tinda_{ses_idx}_{n_states}.pkl', 'wb'))

    # plot the actual cycle
    plot_cycle(best_sequence, fo_density,  significant, new_figure=True)
    plt.title(f'Cycle in Session {ses_idx}')
    plt.savefig(f'{output_dir}/cycle_{ses_idx}_{n_states}.png')
    plt.savefig(f'{output_dir}/cycle_{ses_idx}_{n_states}.svg')
    plt.show()

    return cyc_strength


def test_sign_cycle(ses_idx: int, n_states: int, test_group: bool = True):

    null_model = np.load(f"{output_dir}/null_model_{ses_idx}_{n_states}.npy")
    observed = np.load(f"{output_dir}/observed_cycle_strength_{ses_idx}_{n_states}.npy")

    null_model = np.asarray(null_model)
    observed = np.asarray(observed)

    n_perm = null_model.shape[0]

    if test_group:
        # null_model expected shape: (n_perm, n_samples) OR (n_perm,) depending on what you saved
        if null_model.ndim == 2:
            group_null = null_model.mean(axis=1)
        else:
            group_null = null_model

        group_observed = observed.mean() if observed.ndim > 0 else float(observed)

        # one-sided: observed > null
        p = (np.sum(group_null >= group_observed) + 1) / (len(group_null) + 1)

        plt.hist(group_null, bins=30)
        plt.axvline(group_observed, color="red", linewidth=2, label="observed")
        plt.legend()
        plt.savefig(f"{output_dir}/permutations_cycle_strength_group_{ses_idx}_{n_states}.svg")
        plt.show()

        return p

    else:
        # subject-level p-values (requires null_model shape: (n_perm, n_samples))
        if null_model.ndim != 2:
            raise ValueError("For subject-level p-values, null_model must be shape (n_perm, n_samples).")

        if observed.ndim != 1 or observed.shape[0] != null_model.shape[1]:
            raise ValueError("Observed must be shape (n_samples,) matching null_model second dimension.")

        p_sub = (np.sum(null_model >= observed[None, :], axis=0) + 1) / (n_perm + 1)
        
        return p_sub


# Helper functions (move to utils)
def concatenate_sessions_per_patient(nses, n_states, threed_array=False):
    """
    Load HMM state probabilities for all sessions and reorganize per patient.

    Returns:
        If threed_array=False:
            List of arrays per patient, concatenated across sessions.
        If threed_array=True:
            List of lists: per patient → per session arrays.
    """
    all_sessions = []

    for session_idx in range(nses):
        fp = Path(f"{hmm_dir}/states_{session_idx}_{n_states}.pkl")
        with open(fp, "rb") as f:
            state_probs = pickle.load(f)

        all_sessions.append(state_probs)

    # Reorganise: session-major -> patient-major
    per_patient_session = list(zip(*all_sessions))

    if threed_array:
        return [
            [np.asarray(sess) for sess in patient_sessions]
            for patient_sessions in per_patient_session
        ]
    else:
        return [
            np.concatenate(patient_sessions, axis=0)
            for patient_sessions in per_patient_session
        ]


def permute_state_time_course(state_time_course: np.ndarray, n_states: int):

    # Permute state labels within each patient and each session
    permuted_state_time_course = [permute_state_sequences(stc, n_states) for stc in state_time_course]

    # TINDA
    fo_density, _, _ = tinda(permuted_state_time_course)

    # Best sequence
    best_sequence = optimise_sequence(fo_density)

    # Angles
    angleplot = circle_angles(best_sequence)

    # normalised FO asymmetry
    asym = compute_asym(fo_density)

    # Cycle strength (normalised)
    cyc_strength = compute_cycle_strength(angleplot, asym, relative=True)

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


def compute_asym(fo_density):
    denom = np.mean(fo_density, axis=2, keepdims=True)
    return np.squeeze((fo_density[:,:,0:1,:] - fo_density[:,:,1:2,:]) / denom)