"""Cycle parameter analysis.

Loads cycle strength and cycle duration for each session and patient and
runs regression models to predict baseline symptoms and symptom improvement.

Author: Carina Forster

Last update: 19/01/2026
"""
import pandas as pd
import numpy as np
from pathlib import Path
import os
import pickle

import statsmodels.formula.api as smf
import statsmodels.api as sm
from scipy.stats import zscore, ttest_1samp

from statsmodels.graphics.regressionplots import plot_partregress
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.ticker as mticker

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42

# linux doesn't have Arial
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 13,
    "legend.fontsize": 12,
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

n_states=8
sess_idx=99

# where are the symptoms stored
csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")

def which_run(nruns: int = 5):

    fe_allruns = [
    pickle.load(open(f"{hmm_dir}/run{i+1}/free_energy.pkl", "rb"))
    for i in range(1,5)
    ]

    which_run = int(np.argmin(fe_allruns))

    return which_run

def load_cycle_parameters():

    with open(f"{hmm_dir}/figures/cycles/cycle_rate_{n_states}/run1/cycle_duration_{sess_idx}_{n_states}.pkl", "rb") as f:
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

    assert df['patient'].nunique() == 70

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", "group", "responder"]:
        df[col] = df[col].astype("category", errors="ignore")

    # get rid of states
    df_state1 = df[df["state"] == 1].copy()

    # calculate mean cycle duration per patient and session
    cycle_mean = [c.mean() for c in cycle_duration]

    # add cycle strength
    df_state1['asym_12'] = state1to2
    df_state1['asym_21'] = state2to1
    df_state1['cycle_strength'] = cycle_strength
    df_state1['cycle_duration'] = cycle_mean
    df_state1['cycle_rate'] = df_state1['cycle_duration'].apply(lambda x: 1/x)
    
    df_state1.to_csv(f'{hmm_dir}/df_includingcycles_{n_states}.csv')

    return df_state1



def load_and_prep_data(n_states, exclude_repeater: bool = False):
    
    # read df including cycle params
    csv_path = Path(f'{hmm_dir}/df_includingcycles_{n_states}.csv')

    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(pd.Series(df['patient']))
    
    # exclude patients that repeated TMS treatment?
    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        print(f'{len(repeater_ids)} patients repeated the treatment')
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].str.contains("R")]
        drop_indices = repeater_positions
        np.save(f'{hmm_dir}/dropped_indices.npy', np.array(drop_indices))

    print(f"Analyzing {df['patient'].nunique()} patients")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # fill in demographic variables
    for col in ["age", "gender", 'responder', 'group', 'years_with_depression']:
        df[col] = df.groupby("patient")[col].transform("first")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", 'responder', 'group', 'gender']:
        df[col] = df[col].astype("category", errors="ignore")

    return df


def analyse_cycle_params(df):

    # drop state (no longer needed)
    df_clean = df.drop(columns=['state'])
    df_clean = df_clean.drop_duplicates(subset=['patient', 'session', 'tms'])

    plt.hist(df_clean['cycle_strength'])

    # zscore and remove outlier
    df_removeoutlier = df_clean[
    (np.abs(zscore(df_clean['cycle_rate'], nan_policy='omit')) < 3) &
    (np.abs(zscore(df_clean['cycle_strength'], nan_policy='omit')) < 3) &
    (np.abs(zscore(df_clean['hads_dep_total'], nan_policy='omit')) < 3)
    ]

    # Mixed model with random slopes for session
    model = smf.mixedlm(
        f'cycle_rate ~ session+tms',
        data=df_removeoutlier,
        groups='patient'
    )
    result = model.fit()
    print(result.summary())

    g = sns.lmplot(
        data=df_removeoutlier,
        x='hads_dep_total',
        y='cycle_strength',
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

    cycle_metrics = ['cycle_strength', 'cycle_rate']

    # does PC predict hads score in session 1
    df_sess1_pre = df_removeoutlier.query("session == 1 and tms == 'pre'")
    model = smf.ols("cycle_rate ~ scale(hads_dep_total) + scale(age) + gender + scale(years_with_depression) + group", data=df_sess1_pre).fit()
    print(model.summary())

    # does PC predict hads score in session 1
    df_sess1_pre = df_removeoutlier.query("session == 1 and tms == 'pre'")
    model = smf.ols("cycle_strength ~ scale(hads_dep_total) + group + scale(age) + gender + scale(years_with_depression)", data=df_sess1_pre).fit()
    print(model.summary())

    # plot figure 2
    plot_cycle_params_baseline_hads(df_sess1_pre, 99, n_states)
    
    return df_removeoutlier


def plot_pc_vs_symptom_change(n_states: int,
    symptom_col='hads_dep_total', 
    covariates=['age', 'gender', 'years_with_depression', 'group']
):
    """
    Test whether TMS-induced PC1/PC2 changes predict symptom change (ΔHADS-D)
    across session intervals (2, 3), controlling for covariates.
    Produces separate regression plots for PC1 and PC2.
    """

    df = analyse_cycle_params(df)

    # --- Data prep ---
    df['session'] = df['session'].astype(int)

    # Function to compute Δ pre–post
    def compute_change(df, var):
        wide = df.pivot_table(index=['patient', 'session'], columns='tms', values=var).reset_index()
        wide[f'{var}_change'] = wide['pre'] - wide['post']
        return wide[['patient', 'session', f'{var}_change']]

    cycle_strength_change = compute_change(df, 'cycle_strength')
    cycle_rate_change = compute_change(df, 'cycle_rate')

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
        cycle_rate_change.merge(cycle_strength_change, on=['patient', 'session'])
        .merge(sym_change[['patient', 'session', 'symptom_change']], on=['patient', 'session'])
    )

    df_cov = df[['patient'] + covariates].drop_duplicates()
    df_merge = df_merge.merge(df_cov, on='patient', how='left').dropna()

    # zscore and remove outlier
    df_clean = df_merge[
    (np.abs(zscore(df_merge['cycle_strength_change'], nan_policy='omit')) < 3) &
     (np.abs(zscore(df_merge['cycle_rate_change'], nan_policy='omit')) < 3) &
    (np.abs(zscore(df_merge['symptom_change'], nan_policy='omit')) < 3)
    ]

    # Baseline to mid of treatment
    df_sess1 = df_clean[df_clean['session'] == 1]
    model = smf.ols("symptom_change ~ scale(cycle_strength_change) + scale(age) + gender + scale(years_with_depression) + group", data=df_sess1).fit()
    print(model.summary())

    # Mid to end of treatment
    df_sess2 = df_clean[df_clean['session'] == 2]
    model = smf.ols("symptom_change ~ scale(cycle_strength_change) + scale(age) + gender + scale(years_with_depression) + group", data=df_sess2).fit()
    print(model.summary())

    # cycle rate

    # Baseline to mid of treatment
    df_sess1 = df_clean[df_clean['session'] == 1]
    model = smf.ols("symptom_change ~ scale(cycle_rate_change) + scale(age) + gender + scale(years_with_depression) + group", data=df_sess1).fit()
    print(model.summary())

    # Mid to end of treatment
    df_sess2 = df_clean[df_clean['session'] == 2]
    model = smf.ols("symptom_change ~ scale(cycle_rate_change) + scale(age) + gender + scale(years_with_depression) + group", data=df_sess2).fit()
    print(model.summary())

    plot_symptom_change_correlation(df_clean, covariates)

    return


def plot_symptom_change_correlation(df, covariates):

    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=(7.1, 4),
        sharey=True,
        gridspec_kw={"wspace": 0.4}
    )

    # -------- Session 1 --------
    df_sess1 = df_clean[df_clean['session'] == 1]
    plot_partregress(
        'symptom_change', 'cycle_rate_change',
        covariates,
        data=df_sess1,
        obs_labels=False,
        ax=ax1
    )
    ax1.set_xlabel("Δ HADS-D")
    ax1.set_ylabel("Δ Cycle rate change")
    ax1.text(
        -0.15, 1.1, "A",
        transform=ax1.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax1.set_title("")

    # -------- Session 2 --------
    df_sess2 = df_clean[df_clean['session'] == 2]
    plot_partregress(
        'symptom_change', 'cycle_rate_change',
        covariates,
        data=df_sess2,
        obs_labels=False,
        ax=ax2
    )
    ax2.set_xlabel("Δ HADS-D")
    ax2.set_ylabel("")  # avoid duplicate label
    ax2.text(
        -0.15, 1.1, "B",
        transform=ax2.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax2.set_title("")

    # -------- Styling --------
    for ax in (ax1, ax2):
        for spine in ['top', 'right']:
            ax.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/symptom_change_cycle_rate.png', dpi=300, bbox_inches='tight')
    plt.show()

    return 


def plot_cycle_params_baseline_hads(df, ses_idx, n_states, covariates=['age', 'gender', 'years_with_depression', 'group'], ):
    """
    Plot cycle dynamics.
    
    Parameters:
    - df: dataframe with cycle features and demographics
    - ses_idx: which session (99 is all sessions concatenated)
    - n_states: how many states
    -covariates: covariates for partial regression plot
    """

    # ---- Colorblind-friendly palette ----
    neg_color = '#56B4E9'  # blue
    pos_color = '#E69F00'  # orange

    # Custom diverging colormap: 
    cmap = LinearSegmentedColormap.from_list('orange_blue', [neg_color, 'white', pos_color])

    # brain state colours for cycle
    brain_state_colors = sns.color_palette("tab20", n_colors=n_states)  
    rgb = np.array(brain_state_colors)        # shape: (n_states, 3)

    # set up figure
    fig = plt.figure(figsize=(7.1, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.4)    

    # Panel A: FO Asymmetry Matrix
    asym, significant = calc_sign_asymmetry(output_dir, ses_idx, n_states)

    # Plot mean asymmetry (mean over sessions and patients)
    mean_asym = asym.mean(axis=-1) 

    # 1 to 10
    state_labels = [str(i) for i in range(1, n_states + 1)]
    ax1 = fig.add_subplot(gs[0, 0])
    ax1 = sns.heatmap(mean_asym, cmap=cmap, center=0, xticklabels=state_labels,
                        yticklabels=state_labels, cbar=False)

    # Add stars for significant edges
    # Get the centers of each square
    for (i, j), val in np.ndenumerate(significant):
        if val:  # only for significant edges
            # Use the heatmap's grid coordinates: row = i, col = j
            x = j + 0.5
            y = i + 0.5  # sns.heatmap inverts y-axis automatically
            ax1.text(
                x, y, "*",
                ha="center",
                va="center",
                color="black",
                fontsize=14,
                fontweight="bold"
            )

    ax1.text(-0.15, 1.5, "A", transform=ax1.transAxes,
         fontsize=20, fontweight="bold", va="top")

    # Move x-axis ticks and label to the top
    ax1.xaxis.tick_top()
    ax1.xaxis.set_label_position('top')
    ax1.set_xlabel('Reference state n', labelpad=10)

    # Y-axis labels: vertical (90 degrees)
    ax1.set_yticklabels(
        ax1.get_yticklabels(),
        rotation=360,
        va="center"
    )

    ax1.set_ylabel('HMM network state m')
    ax1.set_xlabel('Reference state n')

    # Create an inset axis above the heatmap
    cax = ax1.inset_axes([0.3, 1.35, 0.5, 0.04])  # [x, y, width, height]

    vmax = np.nanmax(np.abs(mean_asym))
    vmin = -vmax

    norm = plt.Normalize(vmin=vmin, vmax=vmax)

    cbar = plt.colorbar(
        plt.cm.ScalarMappable(norm=norm, cmap=cmap),
        cax=cax,
        orientation='horizontal'
    )

    # Only show extremes
    cbar.set_ticks([-.07, 0, .07])

    # Format ticks to 2 decimals
    cbar.ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter('%.2f')
    )

    # Title above colorbar
    cbar.ax.set_title('FO asymmetry', pad=8)

    # Panel B: cycle

    # load tinda
    with open(f"{output_dir}/tinda_{sess_idx}_{n_states}.pkl", "rb") as f:
        tinda = pickle.load(f)

    best_sequence = tinda['best_sequence']
    fo_density = tinda['fo_density']

    ax2 = fig.add_subplot(gs[0, 1])

    fig, ax = plot_cycle_axis(
        best_sequence,
        fo_density,
        significant,
        color_scheme=rgb,
        ax=ax2,
        new_figure=False
    )

    ax2.text(-0.15, 1.5, "B", transform=ax2.transAxes,
        fontsize=20, fontweight="bold", va="top")

    # ---- Panel C: Partial regression Cycle strength ~ HADS ----
    ax3 = fig.add_subplot(gs[1,0])
    plot_partregress('cycle_strength', 'hads_dep_total', covariates, data=df, obs_labels=False, ax=ax3)
    ax3.set_xlabel("baseline HADS-D")
    ax3.set_ylabel("baseline cycle strength")
    ax3.text(
        -0.15, 1.15, "C",
        transform=ax3.transAxes,
        fontsize=20,
        fontweight="bold",
        va="top"
    )
    ax3.set_title("")

    
    # ---- Panel D: Partial regression Cycle strength ~ HADS ----
    ax4 = fig.add_subplot(gs[1,1])
    plot_partregress('cycle_rate', 'hads_dep_total', covariates, data=df, obs_labels=False, ax=ax4)
    ax4.set_xlabel("baseline HADS-D")
    ax4.set_ylabel("baseline cycle rate")
    ax4.text(
        -0.15, 1.15, "D",
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

    plt.savefig(f'{output_dir}/baseline_cycle_params_{n_states}.png', dpi=300, bbox_inches='tight')
    plt.savefig(f'{output_dir}/baseline_cycle_params_{n_states}.svg', dpi=300, bbox_inches='tight')
    plt.show()

    return fig


def plot_cycle_axis(
    ordering,
    fo_density,
    edges,
    new_figure=False,
    color_scheme=None,
    ax=None,
):
    import numpy as np
    import matplotlib.pyplot as plt

    # -------------------------------
    # Handle figure / axis
    # -------------------------------
    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    # -------------------------------
    # Default color scheme
    # -------------------------------
    if color_scheme is None:
        color_scheme = plt.cm.tab20(np.linspace(0, 1, len(ordering)))[:, :3]

    # -------------------------------
    # Prepare data
    # -------------------------------
    K = len(ordering)

    if fo_density.ndim == 5:
        fo_density = np.squeeze(fo_density)

    mean_direction = (
        fo_density[:, :, 0, :] - fo_density[:, :, 1, :]
    ).mean(axis=2)

    # reorder states
    ordering = np.roll(ordering[::-1], 1)
    edges = edges[ordering][:, ordering]
    mean_direction = mean_direction[ordering][:, ordering]

    # -------------------------------
    # Correct circular coordinates
    # -------------------------------
    theta = np.linspace(0, -2 * np.pi, K, endpoint=False)
    theta += np.pi / 2  # start at 12 o'clock

    radius = 1
    coords = radius * np.column_stack((np.cos(theta), np.sin(theta)))

    # -------------------------------
    # Plot nodes
    # -------------------------------
    for i in range(K):
        ax.scatter(
            coords[i, 0],
            coords[i, 1],
            s=350,
            color=color_scheme[ordering[i]],
            zorder=3,
        )
        ax.text(
            coords[i, 0],
            coords[i, 1],
            str(ordering[i] + 1),
            ha="center",
            va="center",
            fontsize=14,
            zorder=4,
        )

    # -------------------------------
    # Plot arrows
    # -------------------------------
    for i in range(K):
        for j in range(K):
            if edges[i, j]:
                delta = coords[j] - coords[i]
                dist = np.linalg.norm(delta)
                if dist == 0:
                    continue

                start = coords[i] + 0.12 * delta / dist
                end = coords[j] - 0.12 * delta / dist

                direction = mean_direction[i, j]
                if direction == 0:
                    continue

                src, dst = (start, end) if direction > 0 else (end, start)

                ax.annotate(
                    "",
                    xy=dst,
                    xytext=src,
                    arrowprops=dict(
                        arrowstyle="-|>",
                        lw=1.2,
                        color="k",
                        shrinkA=0,
                        shrinkB=0,
                    ),
                    zorder=2,
                )

    # -------------------------------
    # Final formatting (THIS MATTERS)
    # -------------------------------
    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    return fig, ax



def calc_sign_asymmetry(output_dir, ses_idx, n_states):

    # load asymmetry matrix
    asym = np.load(f'{output_dir}/fo_asymmetry_{ses_idx}_{n_states}.npy')

    # Prepare t-test outputs
    t_vals = np.zeros((n_states, n_states))
    p_vals = np.ones((n_states, n_states))

    # List for Bonferroni count
    tests = []

    # Run t-tests
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue

            vals = asym[i, j, :]

            t, p = ttest_1samp(vals, 0, nan_policy="omit")
            t_vals[i, j] = t
            p_vals[i, j] = p
            tests.append((i, j, p))

    # Bonferroni correction
    n_tests = len(tests)
    alpha_corr = 0.05 / n_tests
    significant = p_vals < alpha_corr

    return asym, significant
