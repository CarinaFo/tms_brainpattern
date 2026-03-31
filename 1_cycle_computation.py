"""Compute cyclical dynamics on HMM state time courses.

This script:
1. Loads HMM state probabilities that are saved for each session.
2. Converts probabilities to hard state assignments for TINDA.
3. Computes fractional-occupancy (FO) asymmetry and cycle strength and saves them to disk.
4. Optionally generates a permutation-based null model by permuting the state time course labels.
5. Tests whether the observed cycle strength is greater than the null on a group level.

Author: Carina Forster
Last update: 31/03/2026

Disclaimer: The logic of this code and all steps have been implemented by the author.
            Generative AI (CHAT GPT 5.4 Business) was used to format the script and add
            docstrings to the main functions. 
            A first code-review to find obvious bugs and inconsistencies was
            done using Codex.

            The author takes full responsibility for the code and the scientific results.

Important:
    Run in the ``osld`` environment on Linux or Windows.
"""
import os

# Restrict BLAS/OpenMP threading before importing NumPy/SciPy.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from pathlib import Path
import pickle

import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_1samp
import time
from joblib import Parallel, delayed

# osl functions
from osl_dynamics.inference import modes
import osl_dynamics.analysis.tinda as td

# check n_CPUs
print(f'we have {os.cpu_count()} CPUs')

# Paths
# -----------------------------
system = "windows"  # "linux" or "windows"

if system == "linux":
    base_dir = Path("/home/carinaf/LabData")
elif system == "windows":
    base_dir = Path("L:")
else:
    raise ValueError("system must be 'windows' or 'linux'")

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

output_dir = Path(f"{hmm_dir}/figures/cycles")
output_dir.mkdir(parents=True, exist_ok=True)

nses=6 # we have 6 sessions
nsubs=70 # and 70 patients

# permute state time course labels and compute cycle strength, compare with observed cycle strength
# cyc_strength_observed = calc_cycle_strength(10, False)
# cyc_strength_null = calc_cycle_strength(10, True)
# test_sign_cycle(99, 10)

def calc_cycle_strength(n_states: int, permute_states: bool, n_cores_avail: int = 1, 
                        n_permutations: int = 1000):
    """
    Compute cycle strength using for each individual in each session.

    Args:
        n_states: integer, HMM brain states
        permute_states: bool, run permutation test, ATTENTION: adjust njobs depending on system
        Neuroserv2: 1000 permutations, 20 cores ~ 2hrs
        n_cores_avail: int, how many cores do we have available?
        n_permutations: int, how many permutations(!!!!)

    Returns:
        numpy array with cycle strengths
    """

    # we load the state time course for each session (stacked in a list of patients, list of sessions, timeseries X states)
    # we use SUBJECT MAJOR ordering, e.g. patient 1 session 1, patient 1 session 2 etc.
    stc = concatenate_sessions_per_patient(nses, n_states)

    assert len(stc) == nsubs
    assert len(stc[0]) == nses

    # flatten the list ( we have now subject X sessions stacked vertically, e.g. patient 1 session 1, patient 1 session 2...)
    stc_filtered = [session for patient in stc for session in patient]

    assert len(stc_filtered) == nsubs*nses

    # 99: all sessions concatenated
    pickle.dump({"stc": stc_filtered}, open(f'{output_dir}/stc_99_{n_states}.pkl', 'wb'))

    # plot FO over all sessions to check if there is a state with very high FO
    fo = modes.fractional_occupancies(stc_filtered)  # shape (n_observations, n_states)
    sns.boxplot(fo)
    plt.xlabel('States')
    plt.ylabel('Fractional Occupancy')
    plt.savefig(f'{output_dir}/fo_99_{n_states}.png')
    plt.savefig(f'{output_dir}/fo_99_{n_states}.svg')
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
        np.save(f'{output_dir}/null_model_99_{n_states}.npy', null_model)

        return null_model

    # apply tinda to stc
    fo_density, _, stats = td.tinda(stc_onoff)
    
    # now compute best sequence
    best_sequence = td.optimise_sequence(fo_density)

    # normalised FO asymmetry
    asym = compute_asym(fo_density)

    # save FO asymmetry to disk
    np.save(f'{output_dir}/fo_asymmetry_99_{n_states}.npy', asym)

    # TODO: @MATS I plot fo asymmetry not mean direction, is that ok?
    # normalizes FO asymmetry over patients
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

    # Bonferroni correction for multiple testing
    n_tests = len(tests)
    alpha_corr = 0.05 / n_tests
    significant = p_vals < alpha_corr

    # Plot mean asym
    mean_asym = asym.mean(axis=-1)

    plt.figure(figsize=(7, 6))
    ax = sns.heatmap(mean_asym, cmap="viridis", center=0)

    # Add stars for significant edges
    for i in range(n_states):
        for j in range(n_states):
            if significant[i, j]:
                ax.text(j + 0.5, i + 0.5, "*",
                        ha="center", va="center",
                        color="black", fontsize=14, fontweight="bold")

    #plt.title(f"FO Asymmetry – Significant Connections in session {ses_idx}")
    plt.tight_layout()
    plt.savefig(f"{output_dir}/fo_asymmetry_99_{n_states}.png")
    plt.savefig(f"{output_dir}/fo_asymmetry_99_{n_states}.svg")
    plt.show()

    # now calculcate cycle strength (over all individuals and sessions)
    angleplot = td.circle_angles(best_sequence)

    # calculate cycle strength (normalised)
    cyc_strength = td.compute_cycle_strength(angleplot, asym, relative=True)

    # save observed cycle strength
    np.save(f'{output_dir}/observed_cycle_strength_99_{n_states}.npy', cyc_strength)
    
    # save TINDA output to disk
    pickle.dump({"fo_density":fo_density, "stats": stats, "best_sequence": best_sequence,
                    "asym": asym, "cycle_strength": cyc_strength, "mean_direction": mean_direction}, 
                open(f'{output_dir}/tinda_99_{n_states}.pkl', 'wb'))

    # plot the actual cycle (only significant edges)
    td.plot_cycle(best_sequence, fo_density,  significant, new_figure=True)
    plt.title('Cycle in Session 99')
    plt.savefig(f'{output_dir}/cycle_99_{n_states}.png')
    plt.savefig(f'{output_dir}/cycle_99_{n_states}.svg')
    plt.show()

    return cyc_strength


def test_sign_cycle(ses_idx: int, n_states: int):
    """Test whether observed cycle strength exceeds the permutation null.

    Parameters
    ----------
    ses_idx : int
        Session index used when saving the observed and null-model outputs.
    n_states : int
        Number of HMM states.

    Returns
    -------
    float or np.ndarray
        Group-level one-sided p-value
    """

    # we load the observed and the null model cycle strength
    null_model = np.load(f"{output_dir}/null_model_{ses_idx}_{n_states}.npy")
    observed = np.load(f"{output_dir}/observed_cycle_strength_{ses_idx}_{n_states}.npy")

    null_model = np.asarray(null_model)
    observed = np.asarray(observed)

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


# Helper functions (TODO: move to utils, add docstrings)
def concatenate_sessions_per_patient(nses, n_states):
    """
    Load HMM state probabilities for all sessions and reorganize per patient.
    Returns:
        List of arrays per patient, concatenated across sessions.
    """
    all_sessions = []

    for session_idx in range(nses):
        fp = Path(f"{hmm_dir}/states_{session_idx}_{n_states}.pkl")
        with open(fp, "rb") as f:
            state_probs = pickle.load(f)

        all_sessions.append(state_probs)

    # Reorganise: session-major -> patient-major
    per_patient_session = list(zip(*all_sessions))

    return [
            [np.asarray(sess) for sess in patient_sessions]
            for patient_sessions in per_patient_session
        ]


def permute_state_time_course(state_time_course: np.ndarray, n_states: int):
    # Permute state labels within each patient and each session
    permuted_state_time_course = [permute_state_sequences(stc, n_states) for stc in state_time_course]

    # TINDA
    fo_density, _, _ = td.tinda(permuted_state_time_course)

    # Best sequence
    best_sequence = td.optimise_sequence(fo_density)

    # Angles
    angleplot = td.circle_angles(best_sequence)

    # normalised FO asymmetry
    asym = compute_asym(fo_density)

    # Cycle strength (normalised)
    cyc_strength = td.compute_cycle_strength(angleplot, asym, relative=True)

    return cyc_strength


def permute_state_sequences(state_sequences: np.ndarray, n_states: int = None):
    perm = np.random.permutation(n_states)  # new label for each original state
    permuted_seq = state_sequences[:, perm]  # apply column permutation
    return permuted_seq


def compute_asym(fo_density):
    denom = np.mean(fo_density, axis=2, keepdims=True)
    return np.squeeze((fo_density[:,:,0:1,:] - fo_density[:,:,1:2,:]) / denom)