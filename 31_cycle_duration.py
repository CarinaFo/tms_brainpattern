"""
Compute cycle duration (van Es, Higgins et al., 2025, Nat. Neuro.)

Loads TINDA output (best cycle sequence) and state time courses,
creates windowed features, then fits a second-level Poisson HMM with
sequential Markov dynamics to estimate cycle duration/rate.

Authors: Mats van Es, Carina Forster
Last update: 31/03/2026

Disclaimer: The logic of this code and all steps have been implemented by the author.
            Generative AI (CHAT GPT 5.4 Business) was used to format the script and add
            docstrings to the main functions. 
            A first code-review to find obvious bugs and inconsistencies was
            done using Codex.

            The author takes full responsibility for the code and the scientific results.


Important: run in osld environment on linux (NEUROSERV2)
"""

import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import pickle
import numpy as np
from pathlib import Path

from osl_dynamics.data import Data
from osl_dynamics.inference import modes
from osl_dynamics.models.hmm_poi import Config, Model

print(f"we have {os.cpu_count()} CPUs")

# Paths defined for NeuroServ2 (linux)
base_dir = Path("/home/carinaf/LabData")

hmm_dir = Path(
    f"{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered"
)
output_dir = Path(f"{hmm_dir}/figures/cycles")
output_dir.mkdir(parents=True, exist_ok=True)

# Configs
n_ses = 6 # sessions
n_subs = 70 # patients
ses_all = 99 # all sessions concatenated
fs = 250 # sampling rate

# Cycle configs

# Window length W:
# van Es et al., use average lifetime in samples (≈64–68 ms at 250Hz => W≈16–17 samples)

K1 = 12
K2 = K1 # as discussed with van Es we use the same state resolution in both HMMs
W = 16
run = 0

# run this per state resolution
# TODO: add to main 

#stc_reorder, data, fo = get_reordered_stc_and_windowed_data(
#    W=W, ses=ses_all, nses=n_ses, K1=K1
#)

#cycle_duration, free_energy, run_used = run_second_level_hmm(
#    data=data,
#    fo=fo,
#    K1=K1,
#    K2=K2,
#    fs=fs,
#    W=W,
#    run=run,
#    out_root=output_dir / f"cycle_rate_{K1}_{K2}",
#)

#print(f"Done. run={run_used+1}, free_energy={free_energy}")


def make_windowed_features(stc_reordered: list[np.ndarray], W: int, K1: int):
    """
    Window the the data and counts states within window
        i_data[i, :] = sum(stc[i:i+W, :], axis=0) for i in range(T-W)
    Returns an osl_dynamics Data object containing one array per recording.
    """
    wdata = []

    for stc in stc_reordered:
        stc = np.asarray(stc, dtype=np.float32)
        T, K = stc.shape # time, states
        if K != K1:
            raise ValueError(f"Expected K1={K1}, got {K}.")
        csum = np.vstack([np.zeros((1, K1), dtype=np.float32), np.cumsum(stc, axis=0)])
        # reduce complexity
        Y = csum[W:T] - csum[0:T-W]  # (T-W, K1)
        wdata.append(Y)

    # sanity check: if rows of X sum to 1, rows of Y sum to ~W
    rs = wdata[123].sum(axis=1)

    assert np.isclose(rs.mean(), W, atol=0.01), "Windowed row sums not close to W"

    return Data(wdata)


def get_reordered_stc_and_windowed_data(
    W: int = 16,
    ses: int = ses_all,
    nses: int = n_ses,
    K1: int = K1,
):
    """
    Load stc and tinda output and reorder state time courses, matching author logic.

    Parameters
    ----------
    W : int
        Sliding window length in samples.
    ses : int
        Session id for stc/tinda pickles (99 usually means all concatenated recordings).
    mode : {"group","sub","ses"}
        Determines which best sequence to use:
          - group: single best sequence for all recordings
          - sub:  one best sequence per subject (index ii//nses)
          - ses:  one best sequence per recording (index ii)
        Only "group" is typically needed for replication unless you have those stored.
    nses : int
        Number of sessions (needed for sub mode).
    K1 : int
        Number of first-level states.

    Returns
    -------
    stc_reorder : list[np.ndarray]
        List of reordered state time courses, one per recording.
    data : Data
        Windowed features, one per recording.
    fo : np.ndarray
        Fractional occupancies computed from the original stc (not reordered).
    """
    stc_path = output_dir / f"stc_{ses}_{K1}.pkl"
    tinda_path = output_dir / f"tinda_{ses}_{K1}.pkl"

    stc_obj = _load_pickle(stc_path)
    stc = stc_obj["stc"] if isinstance(stc_obj, dict) else stc_obj

    td = _load_pickle(tinda_path)

    # Binarize state time course
    stc_onoff = modes.argmax_time_courses(stc)
    fo = modes.fractional_occupancies(stc_onoff)

    # life time to determine window length (decided to use 16 as in van Es, Higgins)
    # mean_lt = np.mean(modes.mean_lifetimes(stc_onoff))

    best_seq = np.asarray(td["best_sequence"])
    stc_reordered = [istc[:, best_seq] for istc in stc]

    # Windowed counts
    data = make_windowed_features(stc_reordered, W=W, K1=K1)

    return stc_reordered, data, fo


def init_log_rates(K1: int, K2: int, seq: np.ndarray, fo_mean: np.ndarray, W: int):
    """
    van Es, Higgins et al.: place first-level states on a circle using seq, place K2 meta-states
    as equally spaced centroids on a circle, then weight by FO and scale by W.

    Returns
    -------
    W_mean : (K2, K1) array
        Expected windowed counts per meta-state (Poisson means).
    """
    disttoplot_manual = np.zeros((K1, 2))
    for i in range(K1):
        temp = np.exp(1j * (i + 3) / K1 * 2 * np.pi)
        disttoplot_manual[seq[i], :] = np.array([np.real(temp), np.imag(temp)])

    circleposition = disttoplot_manual[:, 0] + 1j * disttoplot_manual[:, 1]
    metastateposition = [
        (2 ** -0.5) * np.exp(1j * (np.pi / 2 - i_k2 * 2 * np.pi / K2))
        for i_k2 in range(K2)
    ]

    FOweighting = np.zeros((K2, K1))
    for k1 in range(K1):
        for k2 in range(K2):
            FOweighting[k2, k1] = (
                np.real(circleposition[k1]) * np.real(metastateposition[k2])
                + np.imag(circleposition[k1]) * np.imag(metastateposition[k2])
            )

    FOweighting += 1
    FO_metastate = FOweighting * fo_mean
    FO_metastate = FO_metastate / np.sum(FO_metastate, axis=1)[:, np.newaxis]

    W_mean = W * FO_metastate
    return W_mean


def make_circular_trans_mat(K2: int):
    """High self-transition + forward ring"""
    P = 0.99 * np.eye(K2) + 0.01 * np.diag(np.ones(K2 - 1), 1)
    P[-1, 0] = 0.01
    return P


def run_second_level_hmm(
    data: Data,
    fo: np.ndarray,
    K1: int,
    K2: int,
    fs: int = fs,
    W: int = 16,
    run: int = 0,
    out_root: Path | None = None,
):
    """
    Fit second-level Poisson HMM and compute cycle duration as in van Es, Higgins et al., 2025
    """
    if out_root is None:
        out_root = output_dir / f"cycle_rate_{K1}"

    out_root.mkdir(parents=True, exist_ok=True)

    n_runs = int(np.ceil(K1 / K2))
    if run >= n_runs:
        raise ValueError(f"run={run} out of range; n_runs={n_runs} for K1={K1}, K2={K2}")

    rundir = out_root / f"run{run+1}"
    rundir.mkdir(parents=True, exist_ok=True)

    # rotation of the first-level ordering
    seq = np.roll(np.arange(K1).flatten(), run)

    # Poisson means initialisation
    fo_mean = np.asarray(fo).mean(axis=0) if fo.ndim > 1 else np.asarray(fo)
    W_mean = init_log_rates(K1, K2, seq, fo_mean, W=W)

    Pstructure = make_circular_trans_mat(K2)

    # NOTE: config arg name differs by osl-dynamics version: state_probs_t0 vs initial_state_probs
    try:
        config = Config(
            n_states=K2,
            n_channels=K1,
            sequence_length=200,
            initial_trans_prob=Pstructure,
            state_probs_t0=np.ones(K2) / K2,
            learn_trans_prob=True,
            learn_log_rates=False,
            batch_size=1028,
            learning_rate=0.01,
            n_epochs=1,
            initial_log_rates=np.log(W_mean),
        )
    except TypeError:
        # fallback for older API
        config = Config(
            n_states=K2,
            n_channels=K1,
            sequence_length=200,
            initial_trans_prob=Pstructure,
            initial_state_probs=np.ones(K2) / K2,
            learn_trans_prob=True,
            learn_log_rates=False,
            batch_size=1028,
            learning_rate=0.01,
            n_epochs=1,
            initial_log_rates=np.log(W_mean),
        )

    model = Model(config)

    init_history = model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)
    history = model.fit(data)

    free_energy = model.free_energy(data)

    # posteriors
    alp = model.get_alpha(data)
    _save_pickle(alp, rundir / "alp.pkl")

    # viterbi path: needs to be one-hot coded
    # viterbi_path implemented in osl-dynamcis does not allow for leading dimension
    # osl_dynamics version 
    stc_2ndlevel = get_viterbi_path_safe(model, data)
    _save_pickle(stc_2ndlevel, rundir / "stc_2ndlevel.pkl")

    # cycle duration
    cycle_duration = []
    for i_stc in stc_2ndlevel:
        i_stc = np.asarray(i_stc)
        k_init = int(np.argmax(i_stc[0, :]))
        d = np.diff(i_stc[:, k_init])
        ends = np.where(d == -1)[0]
        dur = np.diff(np.append(0, ends)) / fs
        cycle_duration.append(dur)

    _save_pickle(cycle_duration, rundir / "cycle_duration.pkl")

    model.save(rundir / "model")
    _save_pickle(init_history, rundir / "init_history.pkl")
    _save_pickle(history, rundir / "history.pkl")
    _save_pickle(free_energy, rundir / "free_energy.pkl")
    _save_pickle(model.get_log_rates(), rundir / "log_rates.pkl")

    return cycle_duration, free_energy, run

# TODO: put into config.py before publishing

# Helper functions 
def _load_pickle(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)

def _save_pickle(obj, path: Path):
    with open(path, "wb") as f:
        pickle.dump(obj, f)

def one_hot_np(indices, depth, dtype=np.float32):
    """indices: (n,) ints in [0, depth-1] -> (n, depth) one-hot."""
    indices = np.asarray(indices, dtype=np.int64)
    if indices.ndim != 1:
        raise ValueError(f"indices must be 1D, got shape {indices.shape}")
    oh = np.zeros((indices.size, depth), dtype=dtype)
    oh[np.arange(indices.size), indices] = 1
    return oh

def get_viterbi_path_safe(model, dataset, concatenate=False, one_hot=True, one_hot_dtype=np.float32):
    """Like model.get_viterbi_path(), but robust to log-likelihood extra dimension.

    If one_hot=True, returns one-hot Viterbi state time courses (like osl_dynamics):
      - list of (n_samples_session, n_states) if multiple sessions and concatenate=False
      - (n_samples_total, n_states) if concatenate=True or single session
    If one_hot=False, returns integer state sequence(s):
      - list of (n_samples_session,) or (n_samples_total,)
    """
    import sys
    Pi_0 = model.get_initial_state_probs()
    P = model.get_trans_prob()

    eps = sys.float_info.epsilon
    log_Pi_0 = np.log(Pi_0 + eps)
    log_P = np.log(P + eps)

    sequence_length = model.config.sequence_length
    n_states = model.config.n_states

    def _viterbi_path(x):
        log_B = model.get_log_likelihood(x)

        # Handle possible extra leading singleton dim
        if log_B.ndim == 4:
            if log_B.shape[0] != 1:
                raise ValueError(
                    f"Unexpected 4D log_B shape {log_B.shape} (expected leading singleton)."
                )
            log_B = log_B[0]  # -> (batch, seq_len, n_states)
        elif log_B.ndim != 3:
            raise ValueError(f"Unexpected log_B shape {log_B.shape}")

        batch_size = log_B.shape[0]
        if log_B.shape[1] != sequence_length or log_B.shape[2] != n_states:
            raise ValueError(
                f"log_B shape {log_B.shape} incompatible with "
                f"sequence_length={sequence_length}, n_states={n_states}"
            )

        log_prob = np.empty((batch_size, sequence_length, n_states), dtype=float)
        prev = np.empty((batch_size, sequence_length, n_states), dtype=np.int64)

        # init
        log_prob[:, 0, :] = log_Pi_0[np.newaxis, :] + log_B[:, 0, :]

        # recursion
        for t in range(1, sequence_length):
            p = (
                log_prob[:, t - 1, :][..., np.newaxis]
                + log_P[np.newaxis, ...]
                + log_B[:, t, :][..., np.newaxis]
            )
            log_prob[:, t, :] = np.max(p, axis=-2)
            prev[:, t, :] = np.argmax(p, axis=-2)

        # backtrace
        path = np.empty((batch_size, sequence_length), dtype=np.int64)
        path[:, -1] = np.argmax(log_prob[:, -1, :], axis=-1)
        for t in range(sequence_length - 2, -1, -1):
            path[:, t] = prev[np.arange(batch_size), t + 1, path[:, t + 1]]

        return path  # (batch, seq_len) ints

    dataset = model.make_dataset(dataset)

    viterbi_out = []
    for i in range(len(dataset)):
        sess_parts = []
        for batch in dataset[i]:
            x = batch["data"]  # (batch, seq_len, n_channels)
            # _viterbi_path(x) -> (batch, seq_len)
            # concatenate flattens over batch dimension into 1D (batch*seq_len,)
            vp_int = np.concatenate(_viterbi_path(x))
            sess_parts.append(vp_int)

        sess_path_int = np.concatenate(sess_parts)  # (n_samples_session,)
        if one_hot:
            sess_path = one_hot_np(sess_path_int, n_states, dtype=one_hot_dtype)  # (n_samples_session, n_states)
        else:
            sess_path = sess_path_int

        viterbi_out.append(sess_path)

    if concatenate or len(viterbi_out) == 1:
        viterbi_out = np.concatenate(viterbi_out, axis=0)

    return viterbi_out