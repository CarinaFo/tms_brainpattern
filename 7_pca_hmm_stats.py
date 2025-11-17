# Run PCA on HMM summary statistics and relate them to symptom changes

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
fig_dir = Path(f'{hmm_dir}/figures')

#plot_pc_vs_symptom_change()

# --------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------
def load_and_prep_data(csv_path, n_states=n_states, exclude_repeater: bool = True):
    
    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(df.patient)

    # exclude repeater IDs and very noisy IDs
    exclude_ids = [ "127", "182"]
    removed_ids = [list(unique_ids).index(i) for i in exclude_ids]

    df = df[~df["patient"].isin(exclude_ids)]
    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].str.contains("R")]

    drop_indices = removed_ids + repeater_positions
    
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

    plot_symptom_change(df)

    plot_demographics(df)

    return df


def run_PCA():

    df = load_and_prep_data(csv_path, 8, True)

    metrics = ['fo'] #, 'lt', 'sr', 'intv']

    # Pivot: one row per patient/session/tms, columns = state × metric
    df_wide = df.pivot_table(
        index=['patient', 'session', 'tms'],
        columns='state',
        values=metrics
    )

    # Flatten the MultiIndex column names (e.g., fo_state1, lt_state1, etc.)
    df_wide.columns = [f"{m}_state{s}" for m, s in df_wide.columns]
    df_wide = df_wide.reset_index()

    # Select only HMM feature columns, drop one state for fractional occupancy (sum to 1, creates issues for PCA)
    feature_cols = [c for c in df_wide.columns if c.startswith(('fo_', 'lt_', 'sr', 'intv')) and not c.startswith(f'fo_state{n_states}')]

    X = df_wide[feature_cols].dropna()  # remove NaNs

    # Standardize features before PCA
    X_scaled = StandardScaler().fit_transform(X) # should be shape n_patients * sessions, n_states*4-1

    # Fit PCA
    pca = PCA(n_components=None)  # keep all components initially
    X_pca = pca.fit_transform(X_scaled)

    feature_names = X.columns  # list of metric × state names

    # clean up feature_names
    feature_names = [name.replace("state", "") for name in feature_names]

    # Explained variance
    explained = np.cumsum(pca.explained_variance_ratio_) * 100
    print("Cumulative explained variance (%):", explained)

    # plot variance explained (knee plot)
    plot_variance_explained(explained)

    # plot PC loadings
    plot_loadings(pca, feature_cols, feature_names, 'PC2')

    # save the first 3 PCAs
    pca_df = pd.DataFrame(X_pca[:, :3], columns=[f"PC{i+1}" for i in range(3)])
    df_pca = pd.concat([df_wide.reset_index(drop=True), pca_df], axis=1)

    # add clinical and demographics data to dataframe
    keys = ['patient', 'session', 'tms']

    df_merged = df.merge(
        df_pca[['patient', 'session', 'tms', 'PC1', 'PC2', 'PC3']],
        on=keys,
        how='left'
    )

    # drop state (no longer needed)
    df_clean = df_merged.drop(columns=['state'])
    df_clean = df_clean.drop_duplicates(subset=['patient', 'session', 'tms'])

    # does PC predict hads score in session 1
    df_sess1_pre = df_clean.query("session == 1 and tms == 'pre'")
    model = smf.ols("PC2 ~ dep_hads +  age + gender_3 + years_with_depression", data=df_sess1_pre).fit()
    print(model.summary())

    # plot quick regression plot
    sns.regplot(
        data=df_sess1_pre,
        x='dep_hads', y='PC2'
    )
    plt.xlabel("HADS (Session 1 Pre-TMS)")
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

    return df_clean


def plot_pc_vs_symptom_change(
    symptom_col='dep_hads', 
    control_vars=['age', 'gender_3']
):
    """
    Test whether TMS-induced PC1/PC2 changes predict symptom change (ΔHADS-D)
    across session intervals (2, 3), controlling for covariates.
    Produces separate regression plots for PC1 and PC2.
    """

    df = run_PCA()

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


def plot_loadings(pca, feature_cols, feature_names, pc):

    # Component loadings (which features contribute most)
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f"PC{i+1}" for i in range(pca.n_components_)],
        index=feature_cols
    )

    loadings['feature_names'] = feature_names
    # Wes Anderson–inspired palette (green + purple)
    wes_colors = {
        "pc1":  "#7B9E89",   # muted green
        "pc2":  "#A987B1",   # dusty lilac purple
    }

    if pc == 'PC1':
        c = wes_colors['pc1']
    else:
        c = wes_colors['pc2']

     # Sort by PC2 while keeping feature names aligned
    loadings_pc_sorted = loadings[['feature_names', pc]].sort_values(by=pc)
    plt.figure(figsize=(6,8))
    colors = ['#d73027' if x < 0 else '#4575b4' for x in loadings_pc_sorted[pc]]
    plt.barh(loadings_pc_sorted.feature_names, loadings_pc_sorted[pc], color=colors)
    plt.axvline(0, color='black', linestyle='--')
    plt.xlabel(f"{pc} loadings", fontsize=22, color=c)
    plt.ylabel("Features", fontsize=22)
    # Feature names font size
    plt.yticks(fontsize=18)
    plt.xticks(fontsize=18)
    # Remove gridlines
    plt.grid(False)
    # Remove spines for a clean look
    for spine in ['top', 'right']:
        plt.gca().spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/{pc}_loadings.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_symptom_change_correlation(df, pc, sess_label, beta, pval):

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
    plt.savefig(f'{fig_dir}/PC2_deltahads.png', dpi=300, bbox_inches='tight')
    plt.show()


def plot_symptom_change(df):

    # pivot so pre/post are columns
    wide = df.pivot_table(
        index=['patient', 'responder'],
        columns='session',
        values=['dep_hads', 'anxiety_hads', 'madrs_score', 'hama_score']
    )

    # compute deltas (absolute difference)
    diff = pd.DataFrame({
        'hads_dep': (wide['dep_hads'][1] - wide['dep_hads'][3]),
        'hads_anx': (wide['anxiety_hads'][1] - wide['anxiety_hads'][3]),
        'madrs': (wide['madrs_score'][1] - wide['madrs_score'][3]),
        'hama': (wide['hama_score'][1] - wide['hama_score'][3]),
    }).reset_index()

    # Melt to long format for plotting
    diff_long = diff.melt(id_vars=['patient', 'responder'],
                        value_vars=['hads_dep', 'hads_anx', 'madrs', 'hama'],
                        var_name='scale', value_name='change')

    diff_long['responder_label'] = diff_long.responder.map({0: 'Non-Responder', 1: 'Responder'})

    # Set a colorblind-friendly palette (built into seaborn)
    palette = sns.color_palette("colorblind")
    
    plt.figure(figsize=(8, 4))

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
        palette=palette, dodge=True, alpha=0.6, size=5, linewidth=0.5, 
        edgecolor='gray', 
        legend=False
    )

    # Add horizontal reference line
    plt.axhline(0, color='black', linestyle='--', linewidth=1)

    # Labels and style tweaks
    plt.ylabel("Relative Improvement Session 1 → 3 (%)", fontsize=18)
    plt.xlabel("")
    sns.despine(trim=True)
    plt.grid(axis='y', linestyle=':', linewidth=0.6, alpha=0.6)

    # Move legend to lower right outside plot
    plt.legend(
        loc='lower right',
        bbox_to_anchor=(1.0, 0.0),
        frameon=True,
        shadow=False,
        fontsize=18,
        title_fontsize=18
    )

    plt.tight_layout()
    plt.show()


def plot_demographics(df):

    # Keep one row per patient
    df_subj = (
        df.groupby('patient')
        .agg({
            'responder': 'first',
            'age': 'first',
            'gender': 'first',
            'years_with_depression': 'first',
            'group': 'first',
        })
        .reset_index()
    )

    # Create a new column for plotting
    df_subj['responder_label'] = df_subj['responder'].map({0: 'Non-Responder', 1: 'Responder'})

    plt.figure(figsize=(8,4))
    sns.violinplot(data=df_subj, x='responder_label', y='age', palette='colorblind', inner=None)
    sns.boxplot(data=df_subj, x='responder_label', y='age', width=0.2, palette='colorblind')
    plt.xlabel("")
    plt.ylabel("Age (years)")
    sns.despine()
    plt.tight_layout()
    plt.show()

    group_labels={'research tier': 'group', 'self-reported gender': 'gender'}

    for key, value in group_labels.items():
        plt.figure(figsize=(8,4))

        sns.countplot(data=df_subj, x=value, hue='responder_label', palette='colorblind', legend=False)
        plt.xlabel(key)
        plt.ylabel("Count")
        sns.despine()
        plt.tight_layout()
        plt.show()


    plt.figure(figsize=(8,4))
    sns.boxplot(data=df_subj, x='responder_label', y='years_with_depression',
                         palette='colorblind')
    sns.stripplot(data=df_subj, x='responder_label', y='years_with_depression', 
                palette='colorblind', dodge=True, alpha=0.6)
    plt.xlabel("Responder")
    plt.ylabel("Years with Depression")
    sns.despine()
    plt.tight_layout()
    plt.show()
