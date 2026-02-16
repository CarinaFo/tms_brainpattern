# Run PCA on HMM summary statistics and relate them to symptom changes
import pandas as pd
import numpy as np
from pathlib import Path
import os

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.formula.api as smf
from scipy.stats import zscore

from statsmodels.graphics.regressionplots import plot_partregress
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42

# linux doesn't have Arial
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
})

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

def load_and_prep_data(n_states, exclude_repeater: bool = False):
    
    csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(df.patient)

    print(len(unique_ids))

    # exclude patients that repeated TMS treatment?
    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        print(f'{len(repeater_ids)} patients repeated the treatment')
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].str.contains("R")]
        drop_indices = repeater_positions
        np.save(f'{hmm_dir}/dropped_indices.npy', np.array(drop_indices))

    print(f"Analyzing {df['patient'].nunique()} patients")

    # unique patients AFTER filtering
    patient_ids = df["patient"].unique()
    print(f"Total patients: {len(patient_ids)}")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", 'responder', 'group', 'years_with_depression']:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", 'responder', 'group', 'gender']:
        df[col] = df[col].astype("category", errors="ignore")

    plot_responder_group(df)

    plot_hads_over_sessions(df)

    plot_symptom_change(df)

    plot_demographics(df)

    metrics = ["fo", "sr", "lt", "intv"]

    # Average across sessions
    df_avg = (
        df
        .groupby(["patient", "state"])[metrics]
        .mean()
        .reset_index()
    )

    states = sorted(df_avg["state"].unique())
    n_states = len(states)

    # Layout: 2 rows x 5 columns
    n_rows = 2
    n_cols = 5

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(18, 7),
        constrained_layout=True
    )

    # Mask diagonal
    mask = np.eye(len(metrics), dtype=bool)

    axes = axes.flatten()

    for ax, state in zip(axes, states):
        corr = (
            df_avg[df_avg["state"] == state][metrics]
            .corr(method="spearman")
        )

        sns.heatmap(
            corr,
            mask=mask,
            ax=ax,
            vmin=-1, vmax=1,
            cmap="viridis",
            annot=True, fmt=".2f",
            square=True,
            cbar=False
        )

        ax.set_title(f"State {state}", fontsize=20)
        ax.tick_params(labelsize=18)

    # Add a single shared colorbar
    #cbar_ax = fig.add_axes([0.92, 0.25, 0.015, 0.5])
    #sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(-1, 1))
    #sm.set_array([])
    #fig.colorbar(sm, cax=cbar_ax, label="Spearman ρ")

    plt.savefig(f"{fig_dir}/hmm_summary_stats_correlations.svg")
    plt.savefig(f"{fig_dir}/hmm_summary_stats_correlations.png", dpi=300)
    plt.show()

    return df


def run_PCA(n_states: int, clr: bool = False):

    df = load_and_prep_data(n_states, False)

    df = df[df['group'].isin([1,2,3])]

    metrics = ['fo']

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
    feature_cols = [c for c in df_wide.columns if c.startswith(('fo_', 'lt_', 'sr', 'intv'))]

    if clr:
        feature_cols = [c for c in df_wide.columns if c.startswith(('fo_', 'lt_', 'sr', 'intv'))]

        X = df_wide[feature_cols]

        # centered log ratio
        X = clr(X)

    # select columns
    X = df_wide[feature_cols]

    # Standardize features before PCA
    X_scaled = StandardScaler().fit_transform(X) # should be shape n_patients * sessions, n_states

    # Fit PCA
    pca = PCA(n_components=None)  # keep all components initially
    X_pca = pca.fit_transform(X_scaled)

    feature_names = X.columns  # list of metric × state names

    # clean up feature_names
    feature_names = [name.replace("state", "") for name in feature_names]

    # Explained variance
    explained = np.cumsum(pca.explained_variance_ratio_) * 100
    print("Cumulative explained variance (%):", explained)
    plot_variance_explained(explained)

    # save the first 3 PCAs
    pca_df = pd.DataFrame(X_pca[:, :3], columns=[f"PC{i+1}" for i in range(3)])
    df_pca = pd.concat([df_wide.reset_index(drop=True), pca_df], axis=1)

    # add clinical and demographics data to dataframe
    keys = ['patient', 'session', 'tms']

    df_merged = df.merge(
        df_pca[['patient', 'session', 'tms', 'PC1', 'PC2']],
        on=keys,
        how='left'
    )

    # drop state (no longer needed)
    df_clean = df_merged.drop(columns=['state'])
    df_clean = df_clean.drop_duplicates(subset=['patient', 'session', 'tms'])

    # zscore and remove outlier
    df_removeoutlier = df_clean[
    (np.abs(zscore(df_clean['PC2'], nan_policy='omit')) < 3) &
    (np.abs(zscore(df_clean['hads_dep_total'], nan_policy='omit')) < 3) &
    (np.abs(zscore(df_clean['PC1'], nan_policy='omit')) < 3)
    ]

    # does PC predict hads score in session 1
    df_sess1_pre = df_removeoutlier.query("session == 1 and tms == 'pre'")
    model = smf.ols("PC1 ~ hads_dep_total + age + gender + years_with_depression + group", data=df_sess1_pre).fit()
    print(model.summary())

    # does PC predict hads score in session 1
    df_sess1_pre = df_removeoutlier.query("session == 1 and tms == 'pre'")
    model = smf.ols("PC2 ~ hads_dep_total + age + gender + years_with_depression + group", data=df_sess1_pre).fit()
    print(model.summary())

    # plot figure 2
    plot_pca_baseline_hads(pca, df_sess1_pre, feature_cols, feature_names)

    return df_clean


def plot_pc_vs_symptom_change(n_states: int,
    symptom_col='hads_dep_total', 
    covariates=['age', 'gender', 'years_with_depression', 'group']
):
    """
    Test whether TMS-induced PC1/PC2 changes predict symptom change (ΔHADS-D)
    across session intervals (2, 3), controlling for covariates.
    Produces separate regression plots for PC1 and PC2.
    """

    df = run_PCA(n_states)

    # --- Data prep ---
    df['session'] = df['session'].astype(int)

    # Function to compute Δ pre–post
    def compute_change(df, var):
        wide = df.pivot_table(index=['patient', 'session'], columns='tms', values=var).reset_index()
        wide[f'{var}_change'] = wide['pre'] - wide['post']
        return wide[['patient', 'session', f'{var}_change']]

    pc2_change = compute_change(df, 'PC2')

    # --- Symptom change across sessions ---
    sym_wide = df.pivot_table(index='patient', columns='session', values=symptom_col).reset_index()
    
    # change scores
    sym_wide['sym_change_s1_s2'] = sym_wide[1] - sym_wide[2]
    sym_wide['sym_change_s2_s3'] = sym_wide[2] - sym_wide[3]

    # baseline at start of each interval
    sym_wide['baseline_s1_s2'] = sym_wide[1]
    sym_wide['baseline_s2_s3'] = sym_wide[2]

    # reshape
    sym_change = sym_wide.melt(
        id_vars='patient',
        value_vars=['sym_change_s1_s2', 'sym_change_s2_s3'],
        var_name='session_change',
        value_name='symptom_change'
    )

    baseline_long = sym_wide.melt(
        id_vars='patient',
        value_vars=['baseline_s1_s2', 'baseline_s2_s3'],
        var_name='baseline_type',
        value_name='baseline_symptom'
    )

    # align session numbers
    sym_change['session'] = sym_change['session_change'].str.extract(r's(\d)_s\d').astype(int)
    baseline_long['session'] = baseline_long['baseline_type'].str.extract(r's(\d)_s\d').astype(int)

    # merge dataframes
    df_merge = (
        pc2_change
        .merge(sym_change[['patient', 'session', 'symptom_change']], on=['patient', 'session'])
        .merge(baseline_long[['patient', 'session', 'baseline_symptom']], on=['patient', 'session'])
        )

    df_cov = df[['patient'] + covariates].drop_duplicates()
    df_merge = df_merge.merge(df_cov, on='patient', how='left').dropna()

    # zscore and remove outlier
    df_clean = df_merge[
    (np.abs(zscore(df_merge['PC2_change'], nan_policy='omit')) < 3) &
    (np.abs(zscore(df_merge['symptom_change'], nan_policy='omit')) < 3)
    ]

    # --- Regression & plotting ---
    # Baseline to mid of treatment
    df_sess1 = df_clean[df_clean['session'] == 1]
    model = smf.ols("symptom_change ~ PC2_change + baseline_symptom + age + gender + years_with_depression + group", data=df_sess1).fit()
    print(model.summary())

    # Mid to end of treatment
    df_sess2 = df_clean[df_clean['session'] == 2]
    model = smf.ols("symptom_change ~ PC2_change + baseline_symptom + age + gender + years_with_depression + group", data=df_sess2).fit()
    print(model.summary())

    plot_symptom_change_correlation(df_clean, covariates, n_states)

    # PC2 change and symptom change do not sign. correlate between session 1 and 2

    return


### plotting functions
def plot_symptom_change_correlation(df, covariates, n_states):

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7.1, 4),
        sharey=True,
        gridspec_kw={"wspace": 0.4}
    )

    # -------- Session 1 --------
    df_sess1 = df[df['session'] == 1]
    plot_partregress(
        'PC2_change', 'symptom_change',
        covariates,
        data=df_sess1,
        obs_labels=False,
        ax=ax1
    )
    ax1.set_xlabel("Δ HADS-D")
    ax1.set_ylabel("Δ PC2")
    ax1.text(
        -0.15, 1.1, "a",
        transform=ax1.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax1.set_title("Baseline -> Mid-Treatment", fontsize=13)

    # -------- Session 2 --------
    df_sess2 = df[df['session'] == 2]
    plot_partregress(
        'PC2_change', 'symptom_change',
        covariates,
        data=df_sess2,
        obs_labels=False,
        ax=ax2
    )
    ax2.set_xlabel("Δ HADS-D")
    ax2.set_ylabel("")  # avoid duplicate label
    ax2.text(
        -0.15, 1.1, "b",
        transform=ax2.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax2.set_title("Mid -> Post-Treatment", fontsize=13)

    # -------- Styling --------
    for ax in (ax1, ax2):
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{fig_dir}/symptom_change_PC2FO_{n_states}.png', dpi=300, bbox_inches='tight')
    plt.show()

    return 

def plot_symptom_change(df):

    # pivot so pre/post are columns
    wide = df.pivot_table(
        index=['patient', 'responder'],
        columns='session',
        values=['hads_dep_total', 'hads_anx_total', 'madrs_total', 'hama_total']
    )

    # compute deltas (absolute difference)
    diff = pd.DataFrame({
        'hads_dep': (wide['hads_dep_total'][1] - wide['hads_dep_total'][3]),
        'hads_anx': (wide['hads_anx_total'][1] - wide['hads_anx_total'][3]),
        'madrs': (wide['madrs_total'][1] - wide['madrs_total'][3]),
        'hama': (wide['hama_total'][1] - wide['hama_total'][3]),
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


def plot_responder_group(df):

    patient_demo = df.groupby(['patient']).first()

    label_map = {
        1: 'No response < 20 %',
        2: 'Partial response < 50 %',
        3: 'Response < 80 %',
        4: 'Remission > 80 %'
    }

    # Define the desired order: remission → no response
    order = [
        'Remission > 80 %',
        'Response < 80 %',
        'Partial response < 50 %',
        'No response < 20 %'
    ]

    # Green → red colors (remission → no response)
    colors = ['green', 'yellowgreen', 'orange', 'red']

    counts = (
        patient_demo['tms outcome']
        .map(label_map)
        .value_counts()
        .reindex(order)
    )

    ax = counts.plot(
        kind='bar',
        color=colors
    )

    ax.set_xlabel('TMS Outcome')
    ax.set_ylabel('Number of Patients')

    # Rotate x-axis labels
    plt.xticks(rotation=45, ha='right')

    plt.tight_layout()
    plt.show()


def plot_hads_over_sessions(df):

    df_ses = df.groupby(['patient', 'session']).first()

    plt.figure(figsize=(8, 5))

    sns.violinplot(
        data=df_ses,
        x='session',
        y='hads_dep_total',
        inner='box',      # shows median + IQR
        cut=0             # prevents extending beyond data range
    )

    plt.xlabel('Session')
    plt.ylabel('HADS Depression Score')

    plt.tight_layout()
    plt.show()


def plot_group_effects(df_clean):

    # PC2 caries some meaningful variation
    model = smf.mixedlm(
        "hads_dep_total ~ session*group",
        df_clean,
        groups=df_clean["patient"]
    ).fit()

    model.summary()

    plt.figure(figsize=(6, 4))

    for grp, d in df_clean.groupby('group'):
        mean_d = d.groupby('session')['hads_dep_total'].median().reindex([1,2,3])

        plt.plot(
            mean_d.index,
            mean_d.values,
            marker='o',
            linewidth=2,
            label=f'Group {grp}'
        )

    plt.xlabel('Session')
    plt.xticks([1, 2, 3])
    plt.ylabel('Mean HADS-D (pre-TMS)')
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()

    model = smf.mixedlm(
        "PC2 ~ session*group + group*tms",
        df_clean,
        groups=df_clean["patient"]
    ).fit()

    # keep only PC2, session, group, patient, tms
    df_pp = df_clean[['patient', 'group', 'session', 'tms', 'PC2']].copy()

    # pivot pre/post within subject & session
    wide = (
        df_pp
        .pivot_table(
            index=['patient', 'group', 'session'],
            columns='tms',
            values='PC2'
        )
        .reset_index()
    )

    # compute pre - post difference
    wide['PC2_diff'] = wide['pre'] - wide['post']

    # drop rows where diff can't be computed
    wide = wide.dropna(subset=['PC2_diff'])
        
    plt.figure(figsize=(6, 4))

    # ensure session is discrete and ordered
    wide['session'] = wide['session'].astype(int)

    for grp, d in wide.groupby('group'):
        mean_d = (
            d.groupby('session')['PC2_diff']
            .mean()
            .reindex([1, 2, 3])  # enforce discrete order
        )

        plt.plot(
            mean_d.index,
            mean_d.values,
            marker='o',
            linewidth=2,
            label=f'Group {grp}'
        )

    plt.axhline(0, linestyle='--', linewidth=1)
    plt.xlabel('Session')
    plt.ylabel('Mean PC2 (pre − post)')
    plt.title('Mean pre–post PC2 change by session (raw data)')
    plt.xticks([1, 2, 3])
    plt.legend(title='Group')
    plt.tight_layout()
    plt.show()


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
    ax.set_xticks(np.arange(1, len(explained) + 1))

    ax.text(
        -0.15, 1.1, "A",
        transform=ax.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )

    fig.tight_layout()


def plot_pca_baseline_hads(pca, df, feature_cols, feature_names, covariates=['age', 'gender', 'years_with_depression', 'group']):
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

    # ---- Colorblind-friendly palette ----
    neg_color = '#56B4E9'  # blue
    pos_color = '#E69F00'  # orange

    # ---- Panel A: PC1 loadings ----
    loadings = pd.DataFrame(pca.components_.T, columns=[f"PC{i+1}" for i in range(pca.n_components_)], index=feature_cols)
    loadings['feature_names'] = feature_names
    pc1_sorted = loadings[['feature_names','PC1']]
    fnames_pc1 = [f[3:] for f in pc1_sorted.feature_names]
    colors_pc1 = [neg_color if x<0 else pos_color for x in pc1_sorted.PC1]

    ax1 = fig.add_subplot(gs[0,0])
    ax1.bar(fnames_pc1, pc1_sorted.PC1, color=colors_pc1)
    ax1.axhline(0, color='black', linestyle='--')
    ax1.set_xticklabels(fnames_pc1, rotation=0, ha='right')
    ax1.set_ylim(-0.6, 0.6)  # slightly beyond the min/max to give padding
    ax1.set_yticks([-0.5, 0, 0.5])
    ax1.set_yticklabels([-0.5, 0, 0.5])
    ax1.set_xlabel('HMM states')
    ax1.set_ylabel("PC1 Loadings")
    ax1.text(-0.15, 1.1, "a", transform=ax1.transAxes, fontsize=20, fontweight="bold", va="top")

    # ---- Panel B: PC2 loadings ----
    pc2_sorted = loadings[['feature_names','PC2']]
    fnames_pc2 = [f[3:] for f in pc2_sorted.feature_names]
    colors_pc2 = [neg_color if x<0 else pos_color for x in pc2_sorted.PC2]

    ax2 = fig.add_subplot(gs[0,1])
    ax2.bar(fnames_pc2, pc2_sorted.PC2, color=colors_pc2)
    ax2.axhline(0, color='black', linestyle='--')
    ax2.set_xticklabels(fnames_pc2, rotation=0, ha='right')
    ax2.set_ylim(-0.6, 0.6)
    ax2.set_yticks([-0.5, 0, 0.5])
    ax2.set_yticklabels([-0.5, 0, 0.5])
    ax2.set_xlabel('HMM states')
    ax2.set_ylabel("PC2 Loadings")
    ax2.text(-0.15, 1.1, "b", transform=ax2.transAxes, fontsize=20, fontweight="bold", va="top")

    # ---- Panel D: Partial regression PC1 ~ HADS ----
    ax3 = fig.add_subplot(gs[1, 0])
    plot_partregress('PC1', 'hads_dep_total', covariates, data=df, obs_labels=False, ax=ax3)
    ax3.set_xlabel("baseline HADS-D")
    ax3.set_ylabel("baseline PC1")
    ax3.text(
        -0.15, 1.1, "c",
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
        -0.15, 1.1, "d",
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


def clr(X, eps=1e-10):
    X = np.asarray(X, dtype=float)          # <-- converts DataFrame to numpy
    X = np.clip(X, eps, None)               # avoid log(0)
    gm = np.exp(np.mean(np.log(X), axis=1, keepdims=True))
    return np.log(X / gm)
