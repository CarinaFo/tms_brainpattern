"""Cycle parameter analysis.

Loads cycle strength and cycle duration for each session and patient and
runs regression models to predict baseline symptoms and symptom improvement.

Author: Carina Forster

Last update: 15/01/2026
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore

from statsmodels.graphics.regressionplots import plot_partregress
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42

plt.rcParams.update({
    "font.family": "Arial",  # Nature preference
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# run in base python (3.12)
system='linux'

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/LabData')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:")
else:
    "No available system path defined *windows* or *linux*"

# ------------ Directories -------------#
# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

output_dir = Path(f"{hmm_dir}/figures/cycles")

n_states=10
sess_idx=99

# where are the symptoms stored
csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

def load_cycle_parameters():

    with open(f"{rundir}/cycle_duration.pkl", "rb") as f:
        cycle_duration = pickle.load(f)

    # load tinda
    with open(f"{output_dir}/tinda_{sess_idx}_{n_states}.pkl", "rb") as f:
        tinda = pickle.load(f)

    cycle_strength = tinda['cycle_strength']

    # load asymmetry matrix
    asym = tinda['asym']
    
    return asym, cycle_strength, cycle_duration


def add_cycle_parameters_to_df():
    
    # asymmetry between state 1 and 2
    state1to2 = asym[0,1,:]
    state2to1 = asym[1,0,:]

    # load behavioural data
    df = pd.read_csv(csv_path)

    print(f"Analyzing {df['patient'].nunique()} patients")

    assert len(df['patient'].nunique()) == 70

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", "group", "responder"]:
        df[col] = df[col].astype("category", errors="ignore")

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

    df_patient = df_state1[((df_state1['session'] == 3) & (df_state1['tms'] == 'pre'))]

    # remove outlier
    z = np.abs((df_state1['cycle_rate'] - df_state1['cycle_rate'].mean()) / df_state1['cycle_rate'].std())
    df_clean = df_state1[z < 3]
    
    df_clean.to_csv(f'{hmm_dir}/df_includingcycles.csv')

    return df_clean

def analyse_cycle_params(df: pd.DataFrame):

    # Mixed model with random slopes for session
    model = smf.mixedlm(
        f'cycle_rate ~ session*tms',
        data=df_clean,
        groups='patient'
    )

    result = model.fit()
    print(result.summary())

    g = sns.lmplot(
        data=df_clean,
        x='hads_dep_total',
        y='cycle_rate',
        col='session',
        row='tms',
        height=3.5,
        aspect=1,
        scatter_kws={'alpha': 0.7},
        line_kws={'color': 'black'},
        ci=95
    )

    g.set_axis_labels('HADS', 'Cycle rate')
    g.set_titles('Session {col_name} | {row_name} TMS')

    plt.tight_layout()
    plt.show()

    cycle_metrics = ['cycle_strength', 'cycle_rate', 'asym_12', 'asym_21']

    for c in cycle_metrics:

        # Boxplots
        plt.figure(figsize=(12, 6))
        sns.boxplot(x="session", y=c, hue="tms", data=df_clean)
        plt.xlabel("Session")
        plt.ylabel(f"{c}")
        plt.legend(title="TMS")
        plt.tight_layout()
        plt.show()

        plt.hist(df_state1[c])
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
            f'{c}~ responder*session',
            data=df_clean,
            groups="patient"
        )

        result = model.fit()
        print(result.summary())

        df_sess1 = df_clean[(df_clean.session == 3) & (df_clean.tms == 'pre')]

        r, p = scipy.stats.pearsonr(df_sess1.age, df_sess1.cycle_rate)
