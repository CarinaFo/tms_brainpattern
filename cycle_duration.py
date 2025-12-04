"""Cycle Duration Analysis.

Load tinda output and calculate cycle duration based on best sequence
(group level) fitting a second HMM on the fixed state sequence

Authors: Mats van Es
         Carina Forster

Last update: 24-11-2025

Important: run in osld-cuda environment
"""

import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy
from pathlib import Path

import statsmodels.formula.api as smf

from osl_dynamics.data import Data
from osl_dynamics.inference import modes
from osl_dynamics.models.hmm_poi import Config, Model

# ------------ Directories -------------#
home_dir = Path("/home/carinaf/tms_mdd")

tp_dir = Path(f"{home_dir}/80patients_newmodels_giles_plots")
hmm_dir = Path(f"{home_dir}/prepared_data_80patients_giles_newmodel")

output_dir = Path(f"{home_dir}/figures/cycles")
output_dir.mkdir(parents=True, exist_ok=True)

n_states=8
nses=6

csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

# ------------ Functions -------------#

def get_best_state_sequence(W: int = 16):
    """Best cycle sequence based on group"""

    # load stc
    with open(f"{hmm_dir}/stc_sessionspatients_{n_states}.pkl", "rb") as f:
        stc_data = pickle.load(f)

    stc = stc_data['stc']

    # drop patients (index 6 has no EEG data)
    drop_indices = [0, 1, 2, 3, 4, 6, 23, 25, 35, 58] # noisy and repeater patients

    stc_filtered = [s for i, s in enumerate(stc) if i not in drop_indices]

    # flatten the list ( we have now subject X sessions stacked vertically, e.g. patient 1 session 1, patient 1 session 2...)
    stc_filtered = [session for patient in stc_filtered for session in patient]

    # assumes state time course is timepoints X states
    n_states_stc = stc_filtered[0].shape[1]
    
    assert n_states == n_states_stc

    with open(f"{hmm_dir}/tinda_{n_states}states.pkl", "rb") as f:
        tinda = pickle.load(f)

    # best sequence for group
    bs = tinda['best_sequence']['group']

    # Calculate FOs
    fo = modes.fractional_occupancies(stc_filtered)

    # here we reorder state sequences based on best sequence
    stc_reorder = [istc[:, bs] for istc in stc_filtered]

    # create windowed data
    wdata = []
    for i_stc in stc_reorder:
        n_times = i_stc.shape[0]
        i_data = np.zeros((n_times - W, n_states))
        for i in range(n_times - W):
            i_data[i, :] = np.sum(i_stc[i : i + W, :], axis=0)
        wdata.append(i_data)
    data = Data(wdata)

    return stc_reorder, data, fo, n_states


def run_second_level_hmm(
    data: Data = None,
    n_runs: int = 5,
    n_states: int = None,
    n_states_second_level: int = None,
    fs: int = 250,
):
    for i_run in range(n_runs):
        rundir = f"{hmm_dir}/run{i_run+1}"
        os.makedirs(rundir, exist_ok=True)

        # Because we reordered the states according to (individualised) bestseq
        # we can use 1-K1 as bestseq
        seq = np.roll(np.arange(n_states).flatten(), 0)
        W_mean = init_log_rates(n_states, n_states_second_level, seq, fo.mean(axis=0))

        Pstructure = 0.99 * np.eye(n_states_second_level) + 0.01 * np.diag(
            np.ones((n_states_second_level - 1)), 1
        )
        Pstructure[-1, 0] = 0.01

        config = Config(
            n_states=n_states_second_level,
            n_channels=n_states,  # first level HMM states
            sequence_length=200,
            initial_trans_prob=Pstructure,
            initial_state_probs=np.ones(n_states_second_level) / n_states_second_level,
            learn_trans_prob=True,
            learn_log_rates=False,
            batch_size=1028,
            learning_rate=0.01,
            n_epochs=1,
            initial_log_rates=np.log(
                W_mean
            ),  # take the natural log (np.log) of the W_mean
        )
        model = Model(config)

        # Initialization and training
        init_history = model.random_state_time_course_initialization(
            data, n_init=3, n_epochs=1
        )
        history = model.fit(data)

        # Want the run with lowest free energy
        free_energy = model.free_energy(data)
        if i_run == 0 or free_energy < best_fe:
            best_fe = deepcopy(free_energy)
            run = i_run

        # State probabilities
        alp = model.get_alpha(data)
        pickle.dump(alp, open(f"{rundir}/alp.pkl", "wb"))

        # Calculate state time course
        viterbi_paths = []

        # Get fitted transition probability matrix
        trans_prob = model.get_trans_prob()
        initial_state_probs = model.get_initial_state_probs()

        for sp in alp:
            # Wrote my own viterbi path function due to version conflicts
            path = viterbi_from_posteriors(sp, trans_prob, initial_state_probs)
            viterbi_paths.append(path)

        # Save Viterbi paths
        pickle.dump(viterbi_paths, open(f"{rundir}/stc_2ndlevel.pkl", "wb"))

        cycle_duration = []

        for i_stc in viterbi_paths:
            # Get the initial dominant state (i.e., the first state in the path)
            dominant_state = i_stc[0]

            # Create binary vector: 1 when in the dominant state, 0 otherwise
            in_state = (i_stc == dominant_state).astype(int)

            # Detect transitions from the the dominant state
            transitions = np.diff(in_state)

            # Find end points of the dominant state (where it transitions out: 1 → 0)
            ends = np.where(transitions == -1)[0]

            # Compute durations between each exit
            durations = (
                np.diff(np.insert(ends, 0, 0)) / fs
            )  # insert start at 0, then divide by sampling rate

            # Append to list
            cycle_duration.append(durations)

        # Save cycle duration
        pickle.dump(cycle_duration, open(f"{rundir}/cycle_duration.pkl", "wb"))

        # Save trained model
        model.save(f"{rundir}/model")

        # Save training history and free energy
        pickle.dump(init_history, open(f"{rundir}/init_history.pkl", "wb"))
        pickle.dump(history, open(f"{rundir}/history.pkl", "wb"))

        free_energy = model.free_energy(data)
        pickle.dump(free_energy, open(f"{rundir}/free_energy.pkl", "wb"))

        # Observation model parameters
        log_rates = model.get_log_rates()
        pickle.dump(log_rates, open(f"{rundir}/log_rates.pkl", "wb"))

    return cycle_duration


def viterbi_from_posteriors(state_probs, trans_mat, init_probs):
    """
    Compute Viterbi path from state posterior probabilities and transition matrix.

    Args:
        state_probs: np.array (T, K), posterior state probabilities
        trans_mat: np.array (K, K), transition probabilities (rows sum to 1)
        init_probs: np.array (K,), initial state probabilities (sum to 1)

    Returns:
        path: np.array (T,), most likely states sequence (integers in [0, K-1])
    """
    T, K = state_probs.shape
    log_trans = np.log(trans_mat + 1e-16)  # add small epsilon to avoid log(0)
    log_init = np.log(init_probs + 1e-16)
    log_obs = np.log(state_probs + 1e-16)

    delta = np.zeros((T, K))  # max log prob of any path that reaches state k at time t
    psi = np.zeros((T, K), dtype=int)  # backpointer array

    # Initialization
    delta[0] = log_init + log_obs[0]

    # Recursion
    for t in range(1, T):
        for k in range(K):
            seq_probs = (
                delta[t - 1] + log_trans[:, k]
            )  # probs from previous states to k
            psi[t, k] = np.argmax(seq_probs)
            delta[t, k] = seq_probs[psi[t, k]] + log_obs[t, k]

    # Backtracking
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in reversed(range(1, T)):
        path[t - 1] = psi[t, path[t]]

    return path


def init_log_rates(
    n_states: int = None,
    n_states_second_level: int = None,
    seq: np.ndarray = None,
    fo: int = np.ndarray,
    W: int = 16,
):
    """Initialize the log rates for the second-level HMM based on the first-level states.

    Parameters
    ----------
    K1 : int
        Number of first-level states.
    K2 : int
        Number of second-level states.
    seq : array-like, shape (K1,)
        Cycle Sequence ("best_seq") of first-level states.
    fo : array-like, shape (K1,)
        Group level fractional occupancy of first-level states.

    Returns
    -------
    W_mean : array-like, shape (K2, K1)
        Initialized log rates for the second-level HMM.
    """
    disttoplot_manual = np.zeros((n_states, 2))
    for i in range(n_states):
        temp = np.exp(1j * (i + 3) / n_states * 2 * np.pi)
        disttoplot_manual[seq[i], :] = np.array([np.real(temp), np.imag(temp)])

    circleposition = disttoplot_manual[:, 0] + 1j * disttoplot_manual[:, 1]
    metastateposition = [
        (2**-0.5) * np.exp(1j * (np.pi / 2 - i_K2 * 2 * np.pi / n_states_second_level))
        for i_K2 in range(n_states_second_level)
    ]

    FOweighting = np.zeros((n_states_second_level, n_states))
    for k1 in range(n_states):
        for k2 in range(n_states_second_level):
            FOweighting[k2, k1] = np.real(circleposition[k1]) * np.real(
                metastateposition[k2]
            ) + np.imag(circleposition[k1]) * np.imag(metastateposition[k2])

    FOweighting += 1
    FO_metastate = FOweighting * fo
    FO_metastate = FO_metastate / np.sum(FO_metastate, axis=1)[:, np.newaxis]
    W_mean = W * FO_metastate
    return W_mean


def plot_cycle_durations_per_session():

    # load tinda
    with open(f"{hmm_dir}/tinda_sessions_concat_{n_states}states.pkl", "rb") as f:
        tinda = pickle.load(f)

    cycle_strength = tinda['cycle_strength']['group']

    # load asymmetry matrix
    asym = tinda['asym']

    # asymmetry between state 1 and 2
    state1to2 = asym[0,1,:]
    state2to1 = asym[1,0,:]

    # load behavioural data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(df.patient)

    # exclude repeater IDs and very noisy IDs
    exclude_ids = [ "127", "182"]

    df = df[~df["patient"].isin(exclude_ids)]
    df = df[~df["patient"].str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", "group", "responder"]:
        df[col] = df[col].astype("category", errors="ignore")

    # create new column
    df['years_with_depression'] = df['age'] - df['age_of_diagnosis']

    # get rid of states
    df_state1 = df[df['state'] == 1]

    # calculate mean cycle duration per patient and session
    cycle_mean = [c.mean() for c in cycle_duration]

    # add cycle strength
    df_state1['asym_12'] = state1to2
    df_state1['asym_21'] = state2to1
    df_state1['cycle_strength'] = cycle_strength
    df_state1['cycle_duration'] = cycle_mean
    df_state1['cycle_rate'] = df_state1['cycle_duration'].apply(lambda x: 1/x)

    # remove outlier
    z = np.abs((df_state1['cycle_rate'] - df_state1['cycle_rate'].mean()) / df_state1['cycle_rate'].std())
    df_clean = df_state1[z < 1.5]

    cycle_metrics = ['cycle_strength', 'cycle_duration', 'asym_12', 'asym_21']

    for c in cycle_metrics:

        # Boxplots
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="session", y=c, hue="tms", data=df_clean)
        plt.xlabel("Session")
        plt.ylabel(f"{c}")
        plt.legend(title="TMS")
        plt.tight_layout()
        plt.show()

        plt.hist(df_clean[c])
        plt.show()

        # Mixed model with random slopes for session
        model = smf.mixedlm(
            f'{c}~ session*tms',
            data=df_clean,
            groups="patient"
        )

        result = model.fit()
        print(result.summary())

        model = smf.mixedlm(
            f'{c}~ dep_hads*session',
            data=df_clean,
            groups="patient"
        )

        result = model.fit()
        print(result.summary())

        df_sess1 = df_state1[(df_state1.session == 1) & (df.tms == 'pre')]

        df_sess1["log_age"] = np.log(df_sess1["age"])

        model = smf.logit(f'responder ~ {c}', data=df_sess1)
        result = model.fit()
        print(result.summary())


# ------------ Main -------------#

stc_reorder, data, fo, n_states = get_best_state_sequence()
cycle_duration = run_second_level_hmm(data, 1, n_states, 10)