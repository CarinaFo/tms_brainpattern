# Run in base python on lucky3 (Python 3.12)

import numpy as np
import pandas as pd
from pathlib import Path

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.formula.api as smf

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf
import statsmodels.api as sm

# --------------------------------------------------
# Configuration
# --------------------------------------------------
home_dir = Path("L:/Lab_LucaC/Carina/")
n_states = 12

tp_dir = Path(f"{home_dir}/80patients_newmodels_giles_plots")
hmm_dir = Path(f"{home_dir}/prepared_data_80patients_giles_newmodel")

csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

def run_PCA_on_transition_matrix(n_sessions: int = 6):

    all_ses = []

    for ses in range(n_sessions):
        # Load TP matrix
        tp_matrix = np.load(f'{tp_dir}/tp_{ses}_{n_states}.npy')
        all_ses.append(tp_matrix)

    transitions = np.array(all_ses) # sessions, patients, states, states

    n_patients = transitions.shape[1]

    # reshape to sesssions*patients, tp_matrix
    X = transitions.reshape(n_sessions*n_patients, n_states*n_states)

    # Get only off-diagonal elements from each 6x6 matrix
    X_off_diag = np.array([mat[~np.eye(n_states, dtype=bool)] for mat in X.reshape(-1, n_states, n_states)])
    # Now shape is (240, 30)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_off_diag)

    pca = PCA(n_components=None)  # or None to keep all components
    X_pca = pca.fit_transform(X_scaled)
    loadings = pca.components_  # shape (3, 36)

    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative variance:", np.cumsum(pca.explained_variance_ratio_))

    # Cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_) * 100  # In %

    # Plot
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(cumulative_variance)+1), cumulative_variance, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA Elbow Plot (Cumulative Variance)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

    for i in range(2):
        mat = np.zeros((n_states, n_states))
        mat[~np.eye(n_states, dtype=bool)] = loadings[i]
        mat[np.eye(n_states, dtype=bool)] = np.nan  # mask diagonal
        plt.figure(figsize=(6, 5))
        sns.heatmap(mat, annot=True, cmap="coolwarm", center=0, 
                    xticklabels=[f"S{j+1}" for j in range(n_states)],
                    yticklabels=[f"S{i+1}" for i in range(n_states)],
                    mask=np.isnan(mat))
        plt.title(f"PCA Component {i+1} Loadings (off-diagonal only)")
        plt.tight_layout()
        plt.show()

    return X, transitions, X_pca


def add_PCA_to_data():

    X, transitions, X_pca = run_PCA_on_transition_matrix()
    
    df = pd.read_csv(csv_path)

    # Define column names T00, T01, ..., T55
    transition_cols = [f"T{i}{j}" for i in range(n_states) for j in range(n_states)]

    # Create DataFrame
    trans_df = pd.DataFrame(X, columns=transition_cols)

    # add patient ID to transition probability dataframe
    idlist = pd.read_csv(f"{hmm_dir}/patients_fitted_for_this_hmm.csv")

    # Assuming we know the order of patients and sessions
    patients = pd.unique(idlist['patient_id'])
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

    # Optional: reorder columns
    trans_df = trans_df[['patient', 'session'] + transition_cols]

    # save dataframe
    trans_df.to_csv(f'{hmm_dir}/transition_probs_{n_states}.csv')

    # Project data onto the first 3 principal components
    X_pca_3 = X_pca[:, :3]  # shape: (n_samples, 3)

    df_pca_scores = pd.DataFrame(X_pca_3, columns=['PC1', 'PC2', 'PC3'])

    # Add PCA columns
    trans_df = pd.concat([trans_df.reset_index(drop=True), df_pca_scores], axis=1)

    # Optional: reorder columns
    trans_df = trans_df[['patient', 'session', 'PC1', 'PC2', 'PC3'] + transition_cols]

    # Assuming trans_df has a column 'session' (0,1,2,...)
    trans_df['df_session'] = (trans_df['session'] // 2) + 1

    trans_df['tms'] = trans_df['session'].apply(lambda x: 'pre' if x % 2 == 0 else 'post')

    # Make sure 'tms' is treated as a categorical variable
    trans_df['session'] = trans_df['session'].astype('category')

    # Plot PC1 and PC2 over time
    sns.catplot(
    data=trans_df,
    x='session',        # x-axis: pre vs post
    y='PC2',        # y-axis: PC2 values
    col='session',    # facet by session
    kind='swarm',     # or 'swarm' if you want non-overlapping dots
    height=4,
    aspect=0.8,
    palette='Set2',
    dodge=False
    )

    df = df[df['state'] == 1]

    df_merged = trans_df.merge(
    df,
    left_on=['patient','df_session','tms'],
    right_on=['patient','session','tms'],
    how='inner'
    )

    df = df_merged

    # session in with regard to TMS 
    df = df.rename(columns={'session_y': 'session'})

    # session independent of TMS
    df = df.rename(columns={'session_x': 'session_0to5'})

    # mean center for variance inflation factor
    df['hads_dep_score_c'] = df['dep_hads'] - df['dep_hads'].mean()
    
    # exclude repeater IDs and very noisy IDs
    exclude_ids = ["144R", "127", "182"]
    df = df[~df["patient"].isin(exclude_ids)]
    df = df[~df["patient"].str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients")

    # propagate demographic vars
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    # cast common categorical columns (optional, but safe)
    for col in ["patient", "session", "tms", "state", "group", "responder", 'session_0to5']:
        df[col] = df[col].astype("category", errors="ignore")

    df['years_with_depression'] = df['age'] - df['age_of_diagnosis']

    formula_noint = "PC2 ~  dep_hads"

    model = smf.mixedlm(
        formula_noint, 
        df,
        groups=df["patient"]
    )

    result = model.fit(reml=True)

    result.summary()

    df_sess1_pre = df.query("session == 1 and tms == 'pre'")
    model = smf.ols("PC2 ~ dep_hads + age + gender_3 + years_with_depression", data=df_sess1_pre).fit()
    print(model.summary())

    plt.figure(figsize=(5,4))
    sns.regplot(
        data=df_sess1_pre,
        x='dep_hads', y='PC2',
        scatter_kws={'s':60, 'alpha':0.7},
        line_kws={'color':'red'}
    )
    plt.xlabel("HADS Depression (Session 1 Pre-TMS)")
    plt.ylabel("PC2 (Session 1 Pre-TMS)")
    plt.tight_layout()
    plt.show()

    return df


def plot_pc2_tms_vs_symptom_change(
    df, symptom_col='dep_hads', plot=True, control_vars=['age', 'gender_3']
):
    """
    Compute regression between TMS-induced PC2 change and symptom change,
    controlling for age and gender.
    """
    df['session'] = df['session'].astype(int)

    # --- Compute ΔPC2 (post-pre) per session ---
    pc2_wide = df.pivot_table(
    index=['patient', 'session'],
    columns='tms',
    values='PC2'
    ).reset_index()
    pc2_wide['PC2_change'] = pc2_wide['pre'] - pc2_wide['post']
    pc2_change = pc2_wide[['patient', 'session', 'PC2_change']]

    pc1_wide = df.pivot_table(
    index=['patient', 'session'],
    columns='tms',
    values='PC1'
    ).reset_index()
    pc1_wide['PC1_change'] = pc1_wide['pre'] - pc1_wide['post']
    pc2_change['PC1_change'] = pc1_wide['PC1_change']

    # --- Compute absolute symptom change ---
    sym_wide = df.pivot_table(index='patient', columns='session', values=symptom_col).reset_index()
    sym_wide['sym_change_s1_s2'] = sym_wide[1] - sym_wide[3]
    sym_wide['sym_change_s2_s3'] = sym_wide[2] - sym_wide[3]
    sym_wide['sym_change_s1_s3'] = sym_wide[1] - sym_wide[3]
    sym_change = sym_wide.melt(
        id_vars='patient',
        value_vars=['sym_change_s1_s2', 'sym_change_s2_s3', 'sym_change_s1_s3'],
        var_name='session_change',
        value_name='symptom_change'
    )
    sym_change['session'] = sym_change['session_change'].str.extract(r's(\d)_s\d').astype(int)

    df_merge = pc2_change.merge(
    sym_change[['patient', 'session','symptom_change']],
    on=['patient', 'session'],
    how='inner'
    )
    df_merge.drop_duplicates()

    df_cov = df[['patient'] + control_vars].drop_duplicates()
    df_merge = df_merge.merge(df_cov, on='patient', how='left').dropna()

    # --- Regression per session ---
    session_results = []
    for sess in sorted(df_merge['session'].unique()):
        df_s = df_merge[df_merge['session'] == sess].dropna(subset=['PC1_change', 'PC2_change', 'symptom_change'])
        
        # Compute interaction term
        df_s['PC1xPC2'] = df_s['PC1_change'] * df_s['PC2_change']
        
        # Define predictors
        X = df_s[['PC1_change', 'PC2_change', 'PC1xPC2'] + control_vars].copy()
        
        # Handle categorical and numeric conversions
        X = pd.get_dummies(X, drop_first=True)
        X = X.apply(pd.to_numeric, errors='coerce')
        
        # Drop missing values (including y)
        data = pd.concat([X, df_s['symptom_change']], axis=1).dropna()
        X = data.drop(columns='symptom_change')
        y = data['symptom_change']
        
        # Add constant
        X = sm.add_constant(X)
        
        # Fit model
        model = sm.OLS(y, X).fit()
        
        print(model.summary())

        # Extract relevant coefficients
        beta_pc1 = model.params.get('PC1_change', float('nan'))
        p_pc1 = model.pvalues.get('PC1_change', float('nan'))
        beta_pc2 = model.params.get('PC2_change', float('nan'))
        p_pc2 = model.pvalues.get('PC2_change', float('nan'))
        beta_int = model.params.get('PC1xPC2', float('nan'))
        p_int = model.pvalues.get('PC1xPC2', float('nan'))
        
        session_results.append({
            'session': sess,
            'beta_PC1': beta_pc1,
            'p_PC1': p_pc1,
            'beta_PC2': beta_pc2,
            'p_PC2': p_pc2,
            'beta_interaction': beta_int,
            'p_interaction': p_int,
            'n': len(df_s)
        })
        
        # Optional plotting: interaction visualization
        if plot:
            plt.figure(figsize=(5,4))
            sns.scatterplot(data=df_s, x='PC2_change', y='symptom_change', hue='PC1_change', palette='coolwarm')
            sns.regplot(data=df_s, x='PC2_change', y='symptom_change', scatter=False, color='red')
            plt.title(f'Session {sess}: ΔPC1×ΔPC2 interaction')
            plt.xlabel("ΔPC2 (pre-post TMS)")
            plt.ylabel(f"Δ {symptom_col} score")
            plt.tight_layout()
            plt.show()

    results_df = pd.DataFrame(session_results)
    return df_merge, results_df


def predict_symptom_change_from_first_session(
    df, symptom_col='dep_hads', control_vars=['age', 'gender_3'], plot=True
):
    """
    Test whether ΔPC1 and ΔPC2 (pre–post in session 1) predict overall symptom change (session 1→3),
    controlling for age and gender.
    """
    df['session'] = df['session'].astype(int)

    # --- 1️⃣ Compute ΔPC1 and ΔPC2 for each session ---
    pcs_wide = (
        df.pivot_table(index=['patient', 'session'], columns='tms', values=['PC1','PC2'])
        .reset_index()
    )
    pcs_wide['PC1_change'] = pcs_wide['PC1']['pre'] - pcs_wide['PC1']['post']
    pcs_wide['PC2_change'] = pcs_wide['PC2']['pre'] - pcs_wide['PC2']['post']
    pcs_wide.columns = ['patient','session','PC1_pre','PC1_post','PC2_pre','PC2_post',
                        'PC1_change','PC2_change']

    # keep only session 1 changes
    pcs_s1 = pcs_wide.loc[pcs_wide['session']==1, ['patient','PC1_change','PC2_change']]

    # --- 2️⃣ Compute overall symptom change (session 1 → 3) ---
    sym_wide = df.pivot_table(index='patient', columns='session', values=symptom_col)
    sym_wide = sym_wide.rename(columns={1:'s1',2:'s2',3:'s3'})
    sym_wide['symptom_change_1to3'] = (sym_wide['s1'] - sym_wide['s3'])/sym_wide['s1']
    sym_13 = sym_wide[['symptom_change_1to3']].reset_index()

    # --- 3️⃣ Merge ΔPCs from session 1 with symptom change and covariates ---
    df_cov = df[['patient'] + control_vars].drop_duplicates()
    df_merge = pcs_s1.merge(sym_13, on='patient', how='inner').merge(df_cov, on='patient', how='left')
    df_merge = df_merge.dropna(subset=['PC1_change','PC2_change','symptom_change_1to3'])

    # --- 4️⃣ Compute interaction term ---
    df_merge['PC1xPC2'] = df_merge['PC1_change'] * df_merge['PC2_change']

    # --- 5️⃣ Prepare regression matrix ---
    X = df_merge[['PC2_change', 'PC1_change', 'PC1xPC2'] + control_vars].copy()
    X = pd.get_dummies(X, drop_first=True)
    X = X.apply(pd.to_numeric, errors='coerce')
    data = pd.concat([X, df_merge['symptom_change_1to3']], axis=1).dropna()

    X = data.drop(columns='symptom_change_1to3')
    y = data['symptom_change_1to3']
    X = sm.add_constant(X)

    # --- 6️⃣ Fit model ---
    model = sm.OLS(y, X).fit()

    # --- 7️⃣ Optional plot ---
    if plot:
        plt.figure(figsize=(5,4))
        sns.scatterplot(data=df_merge, x='PC2_change', y='symptom_change_1to3',
                         palette='coolwarm')
        sns.regplot(data=df_merge, x='PC2_change', y='symptom_change_1to3',
                    scatter=False, color='red')
        plt.title("ΔPC2 (session 1) vs ΔSymptom (1→3)")
        plt.xlabel("ΔPC2 (pre–post TMS, session 1)")
        plt.ylabel(f"Δ {symptom_col} (1 → 3)")
        plt.tight_layout()
        plt.show()

    return df_merge, model.summary()

