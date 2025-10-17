# Run PCA on HMM summary stats

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np
import statsmodels.formula.api as smf
import statsmodels.api as sm
from pathlib import Path

#plotting
import seaborn as sns
import matplotlib.pyplot as plt

# --------------------------------------------------
# Configuration
# --------------------------------------------------
home_dir = Path("L:/Lab_LucaC/Carina/")
n_states = 12

hmm_dir = Path(f"{home_dir}/prepared_data_80patients_giles_newmodel")
csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

# --------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------
def load_and_prep_data(csv_path, n_states=n_states, exclude_repeater: bool = True):
    
    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    # exclude repeater IDs and very noisy IDs
    exclude_ids = ["144R", "127", "182"]
    df = df[~df["patient"].isin(exclude_ids)]
    if exclude_repeater:
        df = df[~df["patient"].str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # propagate demographic vars
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    # cast common categorical columns (optional, but safe)
    for col in ["patient", "session", "tms", "state", "group", "responder"]:
        df[col] = df[col].astype("category", errors="ignore")

    # pivot so pre/post are columns
    wide = df.pivot_table(
        index=['patient', 'responder'],
        columns='session',
        values=['dep_hads', 'anxiety_hads', 'madrs_score', 'hama_score']
    )

    # compute deltas
    diff = pd.DataFrame({
        'hads_dep': (wide['dep_hads'][1] - wide['dep_hads'][3]),#/wide['dep_hads'][1] *100,
        'hads_anx': (wide['anxiety_hads'][1] - wide['anxiety_hads'][3]),#/wide['anxiety_hads'][1] *100,
        'madrs': (wide['madrs_score'][1] - wide['madrs_score'][3]),#/wide['madrs_score'][1] * 100,
        'hama': (wide['hama_score'][1] - wide['hama_score'][3]),#/wide['hama_score'][1] * 100,
    }).reset_index()

    # Melt to long format for plotting
    diff_long = diff.melt(id_vars=['patient', 'responder'],
                        value_vars=['hads_dep', 'hads_anx', 'madrs', 'hama'],
                        var_name='scale', value_name='change')

    diff_long['responder_label'] = diff_long.responder.map({0: 'Non-Responder', 1: 'Responder'})
    # Set a colorblind-friendly palette (built into seaborn)
    palette = sns.color_palette("colorblind")

    plt.figure(figsize=(7, 5))

    # Boxplot with transparent fill and thicker lines
    sns.boxplot(
        data=diff_long,
        x='scale', y='change', hue='responder_label',
        palette=palette, fliersize=0, linewidth=1.5
    )

    # Add jittered points
    sns.stripplot(
        data=diff_long,
        x='scale', y='change', hue='responder_label',
        palette=palette, dodge=True, alpha=0.6, size=5, linewidth=0.5, edgecolor='gray', legend=False
    )

    # Add horizontal reference line
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Labels and style tweaks
    plt.ylabel("Relative Improvement from Session 1 → 3 (%)", fontsize=14)
    plt.xlabel("")
    sns.despine(trim=True)
    plt.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # Move legend to lower right outside plot
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        frameon=True,
        shadow=False,
        fontsize=10,
        title_fontsize=11
    )

    plt.tight_layout()
    plt.show()

    # Keep one row per patient
    df_subj = (
        df.groupby('patient')
        .agg({
            'responder': 'first',
            'age': 'first',
            'gender': 'first',
            'age_of_diagnosis': 'first',
            'age_of_symptom_onset': 'first',
            'group': 'first',
        })
        .reset_index()
    )

    df_subj['years_with_depression'] = df_subj['age'] - df_subj['age_of_diagnosis']

    # Create a new column for plotting
    df_subj['responder_label'] = df_subj['responder'].map({0: 'Non-Responder', 1: 'Responder'})

    sns.set(style="whitegrid", context="talk")

    plt.figure(figsize=(6,5))
    sns.violinplot(data=df_subj, x='responder_label', y='age', palette='colorblind', inner=None)
    sns.boxplot(data=df_subj, x='responder_label', y='age', width=0.2, palette='colorblind')
    plt.xlabel("")
    plt.ylabel("Age (years)")
    sns.despine()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,5))
    sns.countplot(data=df_subj, x='gender', hue='responder_label', palette='colorblind', legend=False)
    plt.xlabel("Gender")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6,5))
    sns.boxplot(data=df_subj, x='responder_label', y='years_with_depression', palette='colorblind')
    sns.stripplot(data=df_subj, x='responder_label', y='years_with_depression', 
                palette='colorblind', dodge=True, alpha=0.6)
    plt.xlabel("Responder")
    plt.ylabel("Years with Depression")
    sns.despine()
    plt.tight_layout()
    plt.show()


    plt.figure(figsize=(6,5))
    sns.countplot(data=df_subj, x='group', hue='responder_label', palette='colorblind', legend=False)
    plt.xlabel("Research Tier")
    plt.ylabel("Count")
    sns.despine()
    plt.tight_layout()
    plt.show()

    return df, diff


def run_PCA():
    metrics = ['fo', 'lt', 'sr', 'intv']

    # Pivot: one row per patient/session/tms, columns = state × metric
    df_wide = df.pivot_table(
        index=['patient', 'session', 'tms'],
        columns='state',
        values=metrics
    )

    # Flatten the MultiIndex column names (e.g., fo_state1, lt_state1, etc.)
    df_wide.columns = [f"{m}_state{s}" for m, s in df_wide.columns]
    df_wide = df_wide.reset_index()

    # Select only HMM feature columns
    feature_cols = [c for c in df_wide.columns if c.startswith(('fo_', 'lt_', 'sr', 'intv')) and not c.startswith(f'fo_state{n_states}')]

    X = df_wide[feature_cols].dropna()  # remove NaNs

    # Standardize features before PCA
    X_scaled = StandardScaler().fit_transform(X)

    # Fit PCA
    pca = PCA(n_components=None)  # keep all components initially
    X_pca = pca.fit_transform(X_scaled)

    feature_names = X.columns  # list of metric × state names

    # Explained variance
    explained = np.cumsum(pca.explained_variance_ratio_) * 100
    print("Cumulative explained variance (%):", explained)

    # Component loadings (which features contribute most)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_cols
    )
    print(loadings.head())

    plt.plot(np.arange(1, len(explained)+1), explained, marker='o')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance (%)')
    plt.title('PCA Scree Plot')
    plt.show()

    pc_to_plot = "PC2"

    plt.figure(figsize=(12,6))
    plt.bar(feature_names, loadings[pc_to_plot])
    plt.xticks(rotation=90)
    plt.ylabel("Loading")
    plt.title(f"PCA Loadings for {pc_to_plot}")
    plt.tight_layout()
    plt.show()

    # save the first 5 PCAs
    pca_df = pd.DataFrame(X_pca[:, :5], columns=[f"PC{i+1}" for i in range(5)])
    df_pca = pd.concat([df_wide.reset_index(drop=True), pca_df], axis=1)

    # add clinical and demographics data to dataframe
    keys = ['patient', 'session', 'tms']

    df_merged = df.merge(
        df_pca[['patient', 'session', 'tms', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5']],
        on=keys,
        how='left'
    )

    df_clean = df_merged.drop(columns=['state'])
    df_clean = df_clean.drop_duplicates(subset=['patient', 'session', 'tms'])

    
    plt.figure(figsize=(8, 5))
    sns.pointplot(
        data=df_clean,
        x='session',
        y='PC2',
        hue='tms',
        dodge=True,
        errorbar='se',  # or 'sd' for standard deviation
        markers='o',
        capsize=0.1
    )

    plt.title("PC2 across Sessions and TMS")
    plt.ylabel("PC2 score")
    plt.xlabel("Session")
    plt.legend(title="TMS phase")
    plt.tight_layout()
    plt.show()

    df_sess1_pre = df_clean.query("session == 1 and tms == 'pre'")
    model = smf.ols("dep_hads ~ PC2", data=df_sess1_pre).fit()
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

    model = smf.ols("dep_hads ~ PC2 * PC1 * session", data=df_clean).fit()
    print(model.summary())

    # Mixed model using PC1
    model = smf.mixedlm("PC2 ~ session + tms + dep_hads*session", 
                        data=df_clean, groups=df_clean["patient"])
    fit = model.fit()
    print(fit.summary())


    return df_clean


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
    sym_wide['symptom_change_1to3'] = sym_wide['s1'] - sym_wide['s3']
    sym_13 = sym_wide[['symptom_change_1to3']].reset_index()

    # --- 3️⃣ Merge ΔPCs from session 1 with symptom change and covariates ---
    df_cov = df[['patient'] + control_vars].drop_duplicates()
    df_merge = pcs_s1.merge(sym_13, on='patient', how='inner').merge(df_cov, on='patient', how='left')
    df_merge = df_merge.dropna(subset=['PC1_change','PC2_change','symptom_change_1to3'])

    # --- 4️⃣ Compute interaction term ---
    df_merge['PC1xPC2'] = df_merge['PC1_change'] * df_merge['PC2_change']

    # --- 5️⃣ Prepare regression matrix ---
    X = df_merge[['PC2_change', 'PC2_change', 'PC1xPC2'] + control_vars].copy()
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

df, diff = load_and_prep_data(csv_path, n_states=n_states)
df_clean = run_PCA()
plot_pc2_tms_vs_symptom_change(df_clean)
predict_symptom_change_from_first_session(df=df_clean)