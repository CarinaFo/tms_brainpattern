"""Run mixed models on FO

Author: Carina Forster

Last update: 16/01/2026
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore

from statsmodels.graphics.regressionplots import plot_partregress
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42

# linux doesn't have Arial
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
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

n_states=10
sess_idx=99

# where are the symptoms stored
csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

def load_and_prep_data(n_states, exclude_repeater: bool = False):
    
    csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(df.patient)

    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", 'responder', 'group', 'years_with_depression']:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", 'responder', 'group', 'gender']:
        df[col] = df[col].astype("category", errors="ignore")

    return df


def analyze_fo_depression(
    subject_col='patient',
    session_col='session',
    tms_col='tms',
    state_col='state',
    fo_col='fo',
    hads_col='hads_dep_total',
    madrs_col='madrs_total',
    states=(1, 2),
    eps=1e-6
):
    """
    Full FO–depression analysis.

    Analyses included:
    A) Mixed model: FO ~ session * TMS + within-subject HADS
    B) Mixed model: HADS ~ FO * TMS + session
    C) Change-score: ΔHADS ~ ΔFO (all sessions)
    D) Change-score: ΔMADRS ~ ΔFO (sessions 1→3 only)

    Returns:
        pandas.DataFrame with all results
    """

    df = load_and_prep_data(10, False)
    df = df.copy()

    # --------------------------------------------------
    # Subset states
    # --------------------------------------------------
    df = df[df[state_col].isin(states)]

    # --------------------------------------------------
    # Logit transform FO
    # --------------------------------------------------
    df['fo_logit'] = np.log((df[fo_col] + eps) / (1 - df[fo_col] + eps))
    df['hads_log'] = np.log(df[hads_col] + 1)
    df['madrs_log'] = np.log(df[madrs_col] + 1)

    results = []

    # ==================================================
    # A) FO ~ HADS (mixed model)
    # ==================================================
    df['hads_mean'] = df.groupby(subject_col)[hads_col].transform('mean')
    df['hads_within'] = df[hads_col] - df['hads_mean']

    for state in states:
        df_sub = df[df[state_col] == state]

        model = smf.mixedlm(
            f"fo_logit ~ C({session_col}) + C({tms_col}) + hads_within",
            df_sub,
            groups=df_sub[subject_col]
        )
        fit = model.fit(reml=False)

        results.append({
            'analysis': 'FO~HADS_mixed',
            'state': state,
            'term': 'hads_within',
            'beta': fit.params['hads_within'],
            'pval': fit.pvalues['hads_within']
        })

    # ==================================================
    # B) HADS ~ FO (mixed model)
    # ==================================================
    for state in states:
        df_sub = df[df[state_col] == state]

        model = smf.mixedlm(
            f"hads_log ~ fo_logit * C({tms_col}) + C({session_col})",
            df_sub,
            groups=df_sub[subject_col]
        )
        fit = model.fit(reml=False)

        results.append({
            'analysis': 'HADS~FO_mixed',
            'state': state,
            'term': 'fo_logit',
            'beta': fit.params['fo_logit'],
            'pval': fit.pvalues['fo_logit']
        })

        inter = f"fo_logit:C({tms_col})[T.post]"
        if inter in fit.params:
            results.append({
                'analysis': 'HADS~FO_mixed',
                'state': state,
                'term': 'fo_logit:TMS',
                'beta': fit.params[inter],
                'pval': fit.pvalues[inter]
            })

    # ==================================================
    # C) ΔHADS ~ ΔFO (change-score; ALL sessions)
    # ==================================================
    for state in states:

        df_state = df[df[state_col] == state]

        wide = (
            df_state
            .pivot_table(
                index=[subject_col],
                columns=session_col,
                values=['fo_logit', hads_col],
                aggfunc='mean'
            )
            .dropna()
        )

        if wide.shape[1] < 4:
            continue

        delta_fo = wide['fo_logit'].iloc[:, -1] - wide['fo_logit'].iloc[:, 0]
        delta_hads = wide[hads_col].iloc[:, -1] - wide[hads_col].iloc[:, 0]

        X = sm.add_constant(delta_fo)
        model = sm.OLS(delta_hads, X).fit()

        results.append({
            'analysis': 'ΔHADS~ΔFO',
            'state': state,
            'term': 'delta_fo',
            'beta': model.params[0],
            'pval': model.pvalues[0]
        })

    # ==================================================
    # D) ΔMADRS ~ ΔFO (sessions 1 → 3 only)
    # ==================================================
    if madrs_col in df.columns:

        df_madrs = df[
            df[madrs_col].notna() &
            df[session_col].isin([1, 3])
        ]

        for state in states:

            df_state = df_madrs[df_madrs[state_col] == state]

            wide = (
                df_state
                .pivot_table(
                    index=[subject_col],
                    columns=session_col,
                    values=['fo_logit', madrs_col],
                    aggfunc='mean'
                )
                .dropna()
            )

            if set(wide.columns.get_level_values(1)) != {1, 3}:
                continue

            delta_fo = wide['fo_logit'][3] - wide['fo_logit'][1]
            delta_madrs = wide[madrs_col][3] - wide[madrs_col][1]

            X = sm.add_constant(delta_fo)
            model = sm.OLS(delta_madrs, X).fit()

            results.append({
                'analysis': 'ΔMADRS~ΔFO',
                'state': state,
                'term': 'delta_fo',
                'beta': model.params[0],
                'pval': model.pvalues[0]
            })

    return pd.DataFrame(results)
