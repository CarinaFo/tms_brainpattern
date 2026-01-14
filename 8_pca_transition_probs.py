#-------------------------------
# Run PCA on transition probability matrix 
# plot first 2 components
# predict symptom improvement based on delta in PC

# Run in base python on lucky3 or Windows (Python 3.12)
import numpy as np
import pandas as pd
from pathlib import Path
import os

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore

import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import plot_partregress
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import gridspec

# --------------------------------------------------
# Set Paths
# --------------------------------------------------
# run in base python (3.12)

system='windows'

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/LabData')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:")
else:
    "No available system path defined *windows* or *linux*"

# where are the HMM summary stats stored
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')

fig_dir = Path(f'{hmm_dir}/figures')

if not fig_dir.exists():
    os.makedirs(fig_dir, exist_ok=True)

n_states = 10 
n_sessions=6

def run_PCA_on_transition_matrix(n_sessions: int, 
                                    n_states: int):
    '''fit PCA on transition probability matrix obtained from HMM'''
    all_ses = []

    for ses in range(n_sessions):
        # Load TP matrix
        tp_matrix = np.load(f'{hmm_dir}/tp_{ses}_{n_states}.npy')
        all_ses.append(tp_matrix)

    transitions = np.array(all_ses) # sessions, patients, states, states

    # save transition probabilities to disk
    np.save(f'{hmm_dir}/transition_probs.npy', transitions)

    # Continue with reshaping
    n_sessions = transitions.shape[0]
    n_patients = transitions.shape[1]
    n_states = transitions.shape[2]

    X = transitions.reshape(n_sessions * n_patients, n_states * n_states)

    # Get only off-diagonal elements (self transition are much larger)
    X_off_diag = np.array([mat[~np.eye(n_states, dtype=bool)] for mat in X.reshape(-1, n_states, n_states)])
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_off_diag)

    pca = PCA(n_components=None)  # keep all components for plotting
    X_pca = pca.fit_transform(X_scaled)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100

    # plot explained variance
    plot_variance_explained(cumulative_variance)

    loadings = pca.components_ # shape n_patients, n_components

    return X, transitions, X_pca, pca, loadings


def add_PCA_to_data(n_states: int):
    '''add PCs to the clinical dataframe'''

    X, transitions, X_pca, pca, loadings = run_PCA_on_transition_matrix(6, n_states=n_states)

    # open clinical dataframe
    csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

    df = pd.read_csv(csv_path)

    print(f"Analyzing {df['patient'].nunique()} patients")

    # Define column names T00, T01, ..., T55
    transition_cols = [f"T{i}{j}" for i in range(n_states) for j in range(n_states)]

    # Create DataFrame from transitions
    trans_df = pd.DataFrame(X, columns=transition_cols)

    # Assuming we know the order of patients and sessions TODO: (unsafe, recode)
    patients = pd.unique(df['patient'])

    n_sessions = transitions.shape[0]
    n_patients = transitions.shape[1]

    # Repeat each patient ID once per session
    patient_ids = np.tile(patients, n_sessions)

    # Session labels (0..n_sessions-1), repeated for each patient
    session_ids = np.repeat(np.arange(n_sessions), n_patients)

    # Create the transition dataframe
    transition_cols = [f"T{i}{j}" for i in range(n_states) for j in range(n_states)]
    trans_df = pd.DataFrame(X, columns=transition_cols)

    # Add patient and session info
    trans_df['patient'] = patient_ids
    trans_df['session'] = session_ids

    # save dataframe to disk
    trans_df.to_csv(f'{hmm_dir}/transition_probs_{n_states}.csv')

    # Project data onto the first 3 principal components
    X_pca_3 = X_pca[:, :3]  # shape: (n_samples, 3)

    df_pca_scores = pd.DataFrame(X_pca_3, columns=['PC1', 'PC2', 'PC3'])

    # Add PCA columns
    trans_df = pd.concat([trans_df.reset_index(drop=True), df_pca_scores], axis=1)

    # reorder columns
    trans_df = trans_df[['patient', 'session', 'PC1', 'PC2', 'PC3'] + transition_cols]

    # Assuming trans_df has a column 'session' (0,1,2,...)
    trans_df['df_session'] = (trans_df['session'] // 2) + 1

    # recode tms
    trans_df['tms'] = trans_df['session'].apply(lambda x: 'pre' if x % 2 == 0 else 'post')

    # Make sure 'tms' is treated as a categorical variable
    trans_df['session'] = trans_df['session'].astype('category')

    # we don't care about states anymore
    df = df[df['state'] == 1]

    # merge transitions with clinical data
    df_combined = trans_df.merge(
    df,
    left_on=['patient','df_session','tms'],
    right_on=['patient','session','tms'],
    how='inner'
    )

    # bad stuff TODO
    df = df_combined

    # session in with regard to TMS 
    df = df.rename(columns={'session_y': 'session'})

    # session independent of TMS
    df = df.rename(columns={'session_x': 'session_0to5'})

    # mean center depression score for variance inflation factor
    df['hads_dep_score_c'] = df['hads_dep_total'] - df['hads_dep_total'].mean()
    
    # make categorical for mixed linear modelling
    for col in ["patient", "session", "tms", "state", "group", "responder", 'session_0to5']:
        df[col] = df[col].astype("category", errors="ignore")

    # zscore and remove outlier
    df_removeoutlier = df[
        (np.abs(zscore(df['PC2'], nan_policy='omit')) < 3) &
        (np.abs(zscore(df['hads_dep_total'], nan_policy='omit')) < 3)
    ]

    # does PC predict hads score in session 1
    df_sess1_pre = df_removeoutlier.query("session == 1 and tms == 'pre'")

    model = smf.ols("PC1 ~ hads_dep_total + age + gender + years_with_depression + group", data=df_sess1_pre).fit()
    print(model.summary())

    # does PC predict hads score in session 1
    df_sess1_pre = df_removeoutlier.query("session == 1 and tms == 'pre'")
    model = smf.ols("PC2 ~ hads_dep_total + age + gender + years_with_depression + group", data=df_sess1_pre).fit()
    print(model.summary())

    plot_pca_heatmap_baseline_hads(pca, loadings, df_sess1_pre)

    return df


def plot_pc_tms_vs_symptom_change(
    symptom_col='hads_dep_total', 
    control_vars=['age', 'gender', 'years_with_depression']
):
    """
    Test whether TMS-induced PC change predicts symptom change (ΔHADS-D)
    across session intervals (1, 2), controlling for covariates.
    Produces separate regression plots for each PC and each session (2 and 3).
    """

    df = add_PCA_to_data(n_states=10)

    # Function to compute Δ pre–post
    def compute_change(df, var):
        wide = df.pivot_table(index=['patient', 'session'], columns='tms', values=var).reset_index()
        wide[f'{var}_change'] = wide['pre'] - wide['post']
        return wide[['patient', 'session', f'{var}_change']]

    pc1_change = compute_change(df, 'PC1')
    pc2_change = compute_change(df, 'PC2')

    # --- Symptom change across sessions ---
    sym_wide = df.pivot_table(index='patient', columns='session', values=symptom_col).reset_index()
    sym_wide['sym_change_s1_s2'] = sym_wide[1] - sym_wide[2]
    sym_wide['sym_change_s2_s3'] = sym_wide[2] - sym_wide[3]

    sym_change = sym_wide.melt(
        id_vars='patient',
        value_vars=['sym_change_s1_s2', 'sym_change_s2_s3'],
        var_name='session_change',
        value_name='symptom_change'
    )
    sym_change['session'] = sym_change['session_change'].str.extract(r's(\d)_s\d').astype(int)

    # --- Merge ---
    df_merge = (
        pc1_change
        .merge(pc2_change, on=['patient', 'session'])
        .merge(sym_change[['patient', 'session', 'symptom_change']], on=['patient', 'session'])
    )

    df_cov = df[['patient'] + control_vars].drop_duplicates()
    df_merge = df_merge.merge(df_cov, on='patient', how='left').dropna()

    # --- Regression & plotting ---
    results = []

    for row, pc in enumerate(['PC1_change', 'PC2_change']):
        for col, (sess_change, sess_label) in enumerate(zip([1, 2], ["Session 1", "Session 2"])):
            df_s = df_merge[df_merge['session'] == sess_change].dropna(subset=['symptom_change'])

            # Design matrix
            X = df_s[[pc] + control_vars].copy()
            X = pd.get_dummies(X, drop_first=True)
            X = sm.add_constant(X)
            y = df_s['symptom_change']

            model = sm.OLS(y, X).fit()

            beta = model.params.get(pc, np.nan)
            pval = model.pvalues.get(pc, np.nan)

            results.append({
                'session': sess_label,
                'predictor': pc,
                'beta': beta,
                'p': pval,
                'n': len(df_s)
            })

        plot_symptom_change_correlation(df_s, pc, sess_label, beta, pval)

    results_df = pd.DataFrame(results)

    return df_merge, results_df


### plotting functions

def plot_variance_explained(explained):

    fig, ax = plt.subplots(figsize=(7.1, 6))

    ax.plot(
        np.arange(1, len(explained) + 1),
        explained,
        marker='o',
        color='darkgrey',
        linewidth=2
    )

    ax.set_xlabel('Principal Component')
    ax.set_ylabel('Cumulative Variance Explained (%)')
    ax.set_xticks([30, 60, 90])

    ax.text(
        -0.15, 1.1, "A",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )

    fig.tight_layout()


def plot_symptom_change_correlation(df, pc, sess_label, beta, pval):
    '''plot delta PC vs delta symptoms'''
    # Wes Anderson–inspired palette (green + purple)
    wes_colors = {
        "pc1":  "#7B9E89",   # muted green
        "pc2":  "#A987B1",   # dusty lilac purple
    }

    if pc[:3] == 'PC1':
        c = wes_colors['pc1']
    elif pc[:3]:
        c = wes_colors['pc2']

    sns.regplot(
        data=df, x=pc, y='symptom_change', scatter_kws={'s': 70},
        line_kws={'color': 'black'}
    )
    plt.title(f"{(sess_label)}", fontsize=22)
    plt.xlabel(f"Δ {pc.split('_')[0]}", fontsize=22, color=c)
    plt.ylabel("Δ depression score", fontsize=22)
    plt.tick_params(labelsize=18)
    # Remove spines for clean style
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    # Annotation: β and p
    xlim = plt.xlim()
    ylim = plt.ylim()
    plt.text(
        xlim[1]*0.8, ylim[1]*-0.5,  # top-right (adjust as needed)
        f"β = {beta:.2f}\n$p$ = {pval:.3f}",
        ha='right', va='bottom',
        fontsize=18,
        bbox=None,
    )

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/PC2_delta_hads_tp.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_pca_heatmap_baseline_hads(pc, loadings, df, n_states, covariates=['age', 'gender', 'years_with_depression', 'group']):
    """
    Plots PCA summary (variance explained + loadings) and regression of PC1/PC2
    against HADS-D (partial regression adjusting for age and gender)
    
    Parameters:
    - pca: fitted PCA object (sklearn)
    - variance_explained: array-like, cumulative variance explained
    - df: dataframe with PC scores and HADS-D, age, gender, session
    - feature_cols: list of features used in PCA (e.g., states)
    - feature_names: names of features for plotting
    - session_filter: which session to plot regression for (default 'pre')
    """
    fig = plt.figure(figsize=(8, 7))
    gs = gridspec.GridSpec(2, 2, hspace=0.4, wspace=0.4)

    pc_1 = loadings[0, :]
    pc_2 = loadings[1, :]

    # ---- Colorblind-friendly palette ----
    neg_color = '#56B4E9'  # blue
    pos_color = '#E69F00'  # orange

    # ---- Panel A: PC1 loadings ----
    mat = np.zeros((n_states, n_states))
    mat[~np.eye(n_states, dtype=bool)] = pc_1
    mat[np.eye(n_states, dtype=bool)] = np.nan  # mask diagonal

    # Custom diverging colormap: red → white → blue
    cmap = LinearSegmentedColormap.from_list('orange_blue', [neg_color, 'white', pos_color])

    ax1 = fig.add_subplot(gs[0, 0])
    
    sns.heatmap(mat, annot=False, cmap=cmap, center=0, fmt=".2f", ax=ax1,
                xticklabels=[f"{j+1}" for j in range(n_states)],
                yticklabels=[f"{i+1}" for i in range(n_states)],
                mask=np.isnan(mat),
                linewidths=0.5,
                linecolor='lightgrey',
                cbar_kws={'shrink':0.8})

    # Overlay grey diagonal
    for j in range(n_states):
        plt.gca().add_patch(plt.Rectangle((j, j), 1, 1, fill=True, color='grey', zorder=2))

    ax1.set_xlabel("To state")
    ax1.set_ylabel("From state")
    ax1.text(-0.15, 1.1, "A  PC1 loadings", transform=ax1.transAxes,
             fontsize=18, fontweight="bold", va="top")
    
    # ---- Panel A: PC2 loadings ----
    mat = np.zeros((n_states, n_states))
    mat[~np.eye(n_states, dtype=bool)] = pc_2
    mat[np.eye(n_states, dtype=bool)] = np.nan  # mask diagonal

    ax2 = fig.add_subplot(gs[0, 1])
    
    sns.heatmap(mat, annot=False, cmap=cmap, center=0, fmt=".2f", ax=ax2,
                xticklabels=[f"{j+1}" for j in range(n_states)],
                yticklabels=[f"{i+1}" for i in range(n_states)],
                mask=np.isnan(mat),
                linewidths=0.5,
                linecolor='lightgrey',
                cbar_kws={'shrink':0.8})

    # Overlay grey diagonal
    for j in range(n_states):
        plt.gca().add_patch(plt.Rectangle((j, j), 1, 1, fill=True, color='grey', zorder=2))

    ax2.set_xlabel("To state")
    ax2.set_ylabel("From state")
    ax2.text(-0.15, 1.1, "B  PC2 loadings", transform=ax2.transAxes,
             fontsize=18, fontweight="bold", va="top")

    # ---- Panel D: Partial regression PC1 ~ HADS ----
    ax3 = fig.add_subplot(gs[1, 0])
    plot_partregress('PC1', 'hads_dep_total', covariates, data=df, obs_labels=False, ax=ax3)
    ax3.set_xlabel("baseline HADS-D")
    ax3.set_ylabel("baseline PC1")
    ax3.text(
        -0.15, 1.1, "C",
        transform=ax3.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax3.set_title("")

    # ---- Panel E: Partial regression PC2 ~ HADS ----
    ax4 = fig.add_subplot(gs[1, 1])
    plot_partregress('PC2', 'hads_dep_total', covariates, data=df, obs_labels=False, ax=ax4)
    ax4.set_xlabel("baseline HADS-D")
    ax4.set_ylabel("baseline PC2")
    ax4.text(
        -0.15, 1.1, "D",
        transform=ax4.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax4.set_title("")

    # Remove top/right spines for all axes
    for ax in fig.axes:
        for spine in ['top','right']:
            ax.spines[spine].set_visible(False)

    return fig