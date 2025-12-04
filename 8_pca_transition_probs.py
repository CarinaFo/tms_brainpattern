#-------------------------------
# Run PCA on transition probability matrix 
# plot first 2 components
# predict symptom improvement based on delta in PC

# Run in base python on lucky3 or Windows (Python 3.12)
import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.api as sm

# --------------------------------------------------
# Set Paths
# --------------------------------------------------
home_dir = Path("L:/Lab_LucaC/Carina/")

tp_dir = Path(f"{home_dir}/80patients_newmodels_giles_plots")
hmm_dir = Path(f"{home_dir}/prepared_data_80patients_giles_newmodel")

fig_dir = Path(f'{hmm_dir}/figures')

def run_PCA_on_transition_matrix(n_sessions: int = 6, 
                                    exclude_repeater: bool =True, 
                                    n_states: int = None):
    '''fit PCA on transition probability matrix obtained from HMM'''
    all_ses = []

    for ses in range(n_sessions):
        # Load TP matrix
        tp_matrix = np.load(f'{tp_dir}/tp_{ses}_{n_states}.npy')
        all_ses.append(tp_matrix)

    transitions = np.array(all_ses) # sessions, patients, states, states

    # add patient ID to transition probability dataframe
    idlist = pd.read_csv(f"{hmm_dir}/patients_fitted_for_this_hmm.csv")

    # Assuming we know the order of patients and sessions
    patients = pd.unique(idlist['patient_id'])

    # exclude repeater IDs and very noisy IDs
    exclude_ids = ["127", "182"]

    if exclude_repeater:
        exclude_ids += [pid for pid in patients if "R" in pid]
    
    # open clinical dataframe
    csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

    df = pd.read_csv(csv_path)

    clinical_info_patients = pd.unique(df['patient'])

    # one patient has no clinical data but EEG (didn't show up anymore after treatment)
    id_to_drop = [item for item in idlist['patient_id'] if item not in clinical_info_patients]

    exclude_ids.append(id_to_drop[0])

    # Identify indices to keep
    keep_mask = ~idlist["patient_id"].isin(exclude_ids)
    keep_indices = np.where(keep_mask)[0]

    # Filter transitions and patient list
    transitions = transitions[:, keep_indices, :, :]
    patients = [p for i, p in enumerate(patients) if i in keep_indices]

    print(f"Analyzing {len(patients)} patients after exclusion")

    # save transition probabilities to disk
    np.save('transition_probs.npy', transitions)

    # Continue with reshaping
    n_sessions = transitions.shape[0]
    n_patients = transitions.shape[1]
    n_states = transitions.shape[2]

    # load asymmetry matrix
    #asym_matrix = np.load(r"L:\Lab_LucaC\Carina\asym_matrix_8states.npy")

    #X = asym_matrix.reshape(n_states * n_states, 402).T

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

    # plot loadings for PC1 and PC2
    plot_loadings(loadings, n_states=n_states, pc='PC2')

    return X, transitions, X_pca


def add_PCA_to_data(n_states: int = None):
    '''add PCs to the clinical dataframe'''

    X, transitions, X_pca = run_PCA_on_transition_matrix(n_states=n_states)

    # open clinical dataframe
    csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

    df = pd.read_csv(csv_path)

    # exclude repeater IDs and very noisy IDs (EEG after source reco very noisy)
    exclude_ids = ["127", "182"]
    df = df[~df["patient"].isin(exclude_ids)]
    df = df[~df["patient"].str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients") # should be 67

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
    df['hads_dep_score_c'] = df['dep_hads'] - df['dep_hads'].mean()
    
    # make categorical for mixed linear modelling
    for col in ["patient", "session", "tms", "state", "group", "responder", 'session_0to5']:
        df[col] = df[col].astype("category", errors="ignore")

    # new variable: years with depression
    df['years_with_depression'] = df['age'] - df['age_of_diagnosis']

    # does PC predict hads score in session 1
    df_sess1_pre = df[(df["session"] == 1) & (df["tms"] == "pre")]

    model = smf.ols("PC1 ~ dep_hads", data=df_sess1_pre).fit()
    print(model.summary())

    model = smf.ols("PC2 ~ dep_hads + age + gender_3 + years_with_depression", data=df_sess1_pre).fit()
    print(model.summary())

    # plot quick regression plot
    sns.regplot(
        data=df_sess1_pre,
        x='dep_hads', y='PC2'
    )
    plt.xlabel("HADS Depression (Session 1 Pre-TMS)")
    plt.ylabel("PC2 (Session 1 Pre-TMS)")
    plt.tight_layout()
    plt.show()

    sns.regplot(
        data=df_sess1_pre,
        x='dep_hads', y='PC1'
    )
    plt.xlabel("HADS Depression (Session 1 Pre-TMS)")
    plt.ylabel("PC1 (Session 1 Pre-TMS)")
    plt.tight_layout()
    plt.show()

    # PC2 caries some meaningful variation

    return df


def plot_pc_tms_vs_symptom_change(
    symptom_col='dep_hads', 
    control_vars=['age', 'gender_3']
):
    """
    Test whether TMS-induced PC change predicts symptom change (ΔHADS-D)
    across session intervals (1, 2), controlling for covariates.
    Produces separate regression plots for each PC and each session (2 and 3).
    """

    df = add_PCA_to_data(n_states=8)

    # --- Data prep ---
    df['session'] = df['session'].astype(int)

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

def plot_variance_explained(variance_explained):
    '''plot variance explained by PCs'''
    # Wes Anderson–inspired palette (green + purple)
    wes_colors = {
        "pc1":  "#7B9E89",   # muted green
        "pc2":  "#A987B1",   # dusty lilac purple
    }

    plt.figure(figsize=(8,6))

    # Plot full curve
    plt.plot(np.arange(1, len(variance_explained)+1), variance_explained, 
            marker='o', color='darkgrey', linewidth=3)

    # Highlight first two components
    plt.scatter(1, variance_explained[0], color=wes_colors["pc1"], s=180, zorder=3, label='PC1')
    plt.scatter(2, variance_explained[1], color=wes_colors["pc2"], s=180, zorder=3, label='PC2')

    # Labels
    plt.xlabel('Number of Principal Components', fontsize=22)
    plt.ylabel('Cumulative Explained Variance (%)', fontsize=22)

    # Tick label sizes
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)

    # Remove spines for clean style
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    # Legend
    plt.legend(fontsize=18, frameon=False)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/pca_hmm_stats_ncomponents_wes_greenpurple.png',
                dpi=300, bbox_inches='tight')
    plt.show()


def plot_loadings(loadings = None, n_states: int = None, pc: str = None):
    '''plot PC loadings for transition matrix'''
    from matplotlib.colors import LinearSegmentedColormap

    # Wes Anderson–inspired palette (green + purple)
    wes_colors = {
        "pc1":  "#7B9E89",   # muted green
        "pc2":  "#A987B1",   # dusty lilac purple
    }

    if pc == 'PC1':
        c = wes_colors['pc1']
        loading_pc = loadings[0,:]
    else:
        c = wes_colors['pc2']
        loading_pc = loadings[1,:]

    mat = np.zeros((n_states, n_states))
    mat[~np.eye(n_states, dtype=bool)] = loading_pc
    mat[np.eye(n_states, dtype=bool)] = np.nan  # mask diagonal

    plt.figure(figsize=(8, 8))

    # Custom diverging colormap: red → white → blue
    cmap = LinearSegmentedColormap.from_list('red_blue', ['#d73027', 'white', '#4575b4'])

    sns.heatmap(mat, annot=True, cmap=cmap, center=0, fmt=".2f",
                xticklabels=[f"{j+1}" for j in range(n_states)],
                yticklabels=[f"{i+1}" for i in range(n_states)],
                mask=np.isnan(mat),
                linewidths=0.5,
                linecolor='lightgrey',
                cbar_kws={'shrink':0.8})

    # Overlay grey diagonal
    for j in range(n_states):
        plt.gca().add_patch(plt.Rectangle((j, j), 1, 1, fill=True, color='grey', zorder=2))

    # Title and ticks
    plt.title(f"{pc} loadings (excluding self-transitions)", fontsize=20, color=c)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18, rotation=0)
    
    plt.tight_layout()
    plt.savefig(f'{fig_dir}/tp_loadings_{pc}.png', dpi=300, bbox_inches='tight')
    plt.show()


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