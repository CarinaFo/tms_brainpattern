import os
import pickle
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat, loadmat
from osl_dynamics.inference import modes
from osl_dynamics.analysis.tinda import *
from osl_dynamics.data import Data
from osl_dynamics.models.hmm_poi import Config, Model
from copy import deepcopy

mode = argv[1]
run = int(argv[2])
hmmdir = "secondlevel_hmm"
os.makedirs(hmmdir, exist_ok=True)
hmmdir = f"{hmmdir}/{mode}"
os.makedirs(hmmdir, exist_ok=True)


nsub=19
nses=6
W=16 # window size - used to compute a smoothed estimate of the local fractional occupancy of the first level states
K2=12 # number of second level states - can be the same as K1 (it was 4 in the original paper)
K1=12 # number of first level states
fs = 250 # sampling frequency


hmmdir = f"{hmmdir}/K{K2}/"
os.makedirs(hmmdir, exist_ok=True)


def init_log_rates(K1, K2, seq, fo):
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
    disttoplot_manual = np.zeros((K1,2))
    for i in range(K1):
        temp = np.exp(1j*(i+3)/K1*2*np.pi)
        disttoplot_manual[seq[i],:] = np.array([np.real(temp),np.imag(temp)])

    circleposition = disttoplot_manual[:,0] + 1j*disttoplot_manual[:,1]
    metastateposition = [(2**-0.5)*np.exp(1j*(np.pi/2-i_K2*2*np.pi/K2)) for i_K2 in range(K2)]

    FOweighting = np.zeros((K2,K1))
    for k1 in range(K1):
        for k2 in range(K2):
            FOweighting[k2,k1] = np.real(circleposition[k1])*np.real(metastateposition[k2]) + \
                np.imag(circleposition[k1])*np.imag(metastateposition[k2])


    FOweighting += 1
    FO_metastate = FOweighting*fo
    FO_metastate = FO_metastate/np.sum(FO_metastate,axis=1)[:,np.newaxis]
    W_mean = W*FO_metastate
    return W_mean


# create windowed state timecourses
stc = pickle.load(open('stc.pkl', 'rb'))
td = pickle.load(open('tinda.pkl', 'rb'))

if mode=='group':
    stc_reorder = [istc[:, td['best_sequence'][mode]] for istc in stc]
elif mode=='sub':
    stc_reorder = [istc[:, td['best_sequence'][mode][ii//nses]] for ii, istc in enumerate(stc)]
elif mode=='ses':
    stc_reorder = [istc[:, td['best_sequence'][mode][ii]] for ii, istc in enumerate(stc)]

fo = modes.fractional_occupancies(stc)

wdata=[]
for i_stc in stc_reorder:
    n_times = i_stc.shape[0]
    i_data=np.zeros((n_times-W, K1))
    for i in range(n_times-W):
        i_data[i,:] = np.sum(i_stc[i:i+W,:], axis=0)
    wdata.append(i_data)
data = Data(wdata)


best_fe=np.inf
n_runs = int(np.ceil(K1/K2))# the meta states can be centered on different first level states
if run>n_runs:
    print(f"run {run} is out of range")
    exit()
    
for i_run in [run]:#range(n_runs):   
    rundir = f"{hmmdir}/run{i_run+1}"
    os.makedirs(rundir, exist_ok=True)

    # because we reordered the states according to (individualised) bestseq, we can use 1-K1 as bestseq
    seq = np.roll(np.arange(K1).flatten(), i_run)
    W_mean = init_log_rates(K1, K2, seq, fo.mean(axis=0))
    
    Pstructure = 0.99*np.eye(K2) + 0.01*np.diag(np.ones((K2-1)),1)
    Pstructure[-1,0] = .01

    config = Config(
        n_states=K2,
        n_channels=K1,
        sequence_length=200,
        initial_trans_prob=Pstructure,
        state_probs_t0=np.ones(K2)/K2,
        learn_trans_prob=True,
        learn_log_rates=False,
        batch_size=1028,
        learning_rate=0.01,
        n_epochs=1,
        initial_log_rates = np.log(W_mean), # take the natural log (np.log) of the W_mean
    )

    model = Model(config)

    #%% Training

    # Initialisation
    init_history = model.random_state_time_course_initialization(data, n_init=3, n_epochs=1)

    # Full training
    history = model.fit(data)

    #free_energy = model.free_energy(data)
    #if i_run==0 or free_energy<best_fe:
    #    best_fe = deepcopy(free_energy)
    #    run = i_run

        # State probabilities
    alp = model.get_alpha(data)
    pickle.dump(alp, open(f"{rundir}/alp.pkl", "wb"))

    # cycle duration
    stc_2ndlevel = model.get_viterbi_path(data)
    pickle.dump(stc_2ndlevel, open(f"{rundir}/stc_2ndlevel.pkl", "wb"))
    
    cycle_duration = []
    for i_stc in stc_2ndlevel:
        k_init = np.argmax(i_stc[0,:])
        d=np.diff(i_stc[:,k_init])
        cycle_duration.append(np.diff(np.append(0, np.where(d==-1)[0]))/fs)
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
