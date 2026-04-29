"""Cycle parameter analysis.

Loads cycle strength and cycle duration for each session and patient and
runs regression models to predict baseline symptoms and symptom improvement.

Author: Carina Forster

Disclaimer: The logic of this code and all steps have been implemented by the author.
            Generative AI (CHAT GPT 5.4 Business) was used to format the script and add
            docstrings to the main functions. 
            A first code-review to find obvious bugs and inconsistencies was
            done using Codex.

            The author takes full responsibility for the code and the scientific results.

Last update: 31/03/2026
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle

import statsmodels.formula.api as smf
from scipy.stats import zscore, ttest_1samp

import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.ticker as mticker

# text editable in inkscape
plt.rcParams['svg.fonttype'] = 'none'

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

# run in base python (3.10)
system='windows'

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

eeg_length_path = Path(
    f"{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/"
    "prepared_data_giles_05Hz_1Hzfiltereddata/eeg_recording_length.csv")

# first levevl hmm states are set to be equal to second level hmm states as discussed with Mats van Es
n_states_level1 = [10]
n_states_level2 = n_states_level1 # same state resolution on both HMM levels

sess_idx=99 # 99: all sessions concatenated

def analyse_cycle_params(n_states_level1: int, robust="HC3"):

    df_cycle = load_and_prep_data(n_states_level1)

    # better for linear modelling
    df_cycle['cycle_strength'] = df_cycle['cycle_strength']*100

    df_clean = (
        df_cycle
        .drop(columns=["state"], errors="ignore")
        .drop_duplicates(subset=["patient", "session", "tms"])
        .copy()
    )

    # outlier removal
    # df_removeoutlier = df_clean[
    #    (np.abs(zscore(df_clean["cycle_rate"].astype(float), nan_policy="omit")) < 3) &
    #    (np.abs(zscore(df_clean["cycle_strength"].astype(float), nan_policy="omit")) < 3) &
    #    (np.abs(zscore(df_clean["hads_dep_total"].astype(float), nan_policy="omit")) < 3)
    #].copy()

    # baseline-only subset
    d0 = df_clean.query("session == 1 and tms == 'pre'").copy()

    m_rate = smf.ols("np.log(cycle_rate) ~ hads_dep_total + age + C(gender)", data=d0).fit(cov_type=robust)
    m_strength = smf.ols("cycle_strength ~ hads_dep_total + age + C(gender)", data=d0).fit(cov_type=robust)

    print(m_rate.summary())
    print(m_strength.summary())

    # control analysis: include FO as regressor (anterior DMN)
    m_rate_control = smf.ols("np.log(cycle_rate) ~ hads_dep_total + fo + age + C(gender)", data=d0).fit(cov_type=robust)
    m_strength_control = smf.ols("cycle_strength ~ hads_dep_total + fo + age + C(gender)", data=d0).fit(cov_type=robust)

    print(m_rate_control.summary())
    print(m_strength_control.summary())
    # multicollinearity warnings

    # check correlations
    np.corrcoef(d0['cycle_rate'], d0['cycle_strength'])
    np.corrcoef(d0['cycle_rate'], d0['fo'])
    np.corrcoef(d0['cycle_strength'], d0['fo'])
    
    plot_cycle_params_baseline_hads(d0, ses_idx=sess_idx, n_states_level1=n_states_level1)

    return df_clean


def plot_pc_vs_symptom_change(n_states_level1: int,
    symptom_col='hads_dep_total', 
    covariates=['age', 'gender'] # control analysis: include FO as regressor
):
    """
    Test whether TMS-induced PC1/PC2 changes predict symptom change (ΔHADS-D)
    across session intervals (1, 2), controlling for covariates.
    Outputs separate regression plots for PC1 and PC2.
    """

    df = analyse_cycle_params(n_states_level1)

    # --- Data prep ---
    df["session"] = df["session"].astype(int)

    def compute_change(df, var):
        wide = df.pivot_table(index=["patient", "session"], columns="tms", values=var).reset_index()
        wide[f"{var}_change"] = wide["pre"].astype(float) - wide["post"].astype(float)
        return wide[["patient", "session", f"{var}_change"]]

    cycle_strength_change = compute_change(df, "cycle_strength")
    cycle_rate_change = compute_change(df, "cycle_rate")

    # Symptoms wide (one row per patient)
    sym_wide = (
        df.pivot_table(index="patient", columns="session", values=symptom_col)
        .reset_index()
        .rename(columns={1: "s1", 2: "s2", 3: "s3"})
    )

    # --- Merge: cycle change is long (patient, session); symptoms are wide (patient) ---
    df_merge = (
        cycle_rate_change
        .merge(cycle_strength_change, on=["patient", "session"], how="inner")
        .merge(sym_wide[["patient", "s1", "s2", "s3"]], on="patient", how="inner")
    )

    df_cov = df[["patient"] + covariates].drop_duplicates()
    df_merge = df_merge.merge(df_cov, on="patient", how="left").dropna()

    # Outlier removal (only cycle metrics, as you had)
    #df_clean = df_merge[
    #    (np.abs(zscore(df_merge["cycle_strength_change"], nan_policy="omit")) < 3) &
    #    (np.abs(zscore(df_merge["cycle_rate_change"], nan_policy="omit")) < 3)
    #].copy()

    df_clean = df_merge

    # Build baseline + next symptom columns depending on session
    df_clean["sym_base"] = np.where(df_clean["session"] == 1, df_clean["s1"], df_clean["s2"])
    df_clean["sym_next"] = np.where(df_clean["session"] == 1, df_clean["s2"], df_clean["s3"])

    df_sess1 = df_clean[df_clean["session"] == 1].copy()

    model_s1 = smf.ols(
        "sym_next ~ cycle_rate_change + cycle_strength_change + sym_base + age + C(gender)",
        data=df_sess1
    ).fit(cov_type='HC3')

    print(model_s1.summary())

    df_sess2 = df_clean[df_clean["session"] == 2].copy()

    model_s2 = smf.ols(
        "sym_next ~ cycle_rate_change + cycle_strength_change + sym_base + age + C(gender)",
        data=df_sess2
    ).fit(cov_type='HC3')

    print(model_s2.summary())

    models = {1: model_s1, 2: model_s2}

    plot_two_session_cycle_regression(
        df_clean=df_clean,
        models=models,
        savepath=f"{output_dir}/symptom_change_cycle_rate_strength_pred"
    )

    return


def plot_two_session_cycle_regression(
    df_clean,
    models,  # dict: {1: model_for_session1, 2: model_for_session2}
    metrics=("cycle_rate_change", "cycle_strength_change"),
    figsize=(12, 10),
    n_grid=200,
    savepath=None,
    y_label="Predicted next-session symptom severity",
):
    """
    2x2 plot:
      rows = session (1, 2) [i.e., 1->2 and 2->3]
      cols = cycle metrics (rate, strength)

    Each panel:
      - scatter observed sym_next vs x_metric
      - predicted mean + 95% CI from session-specific OLS model:
          sym_next ~ rate_change + strength_change + sym_base + age + C(gender)
      - hold the *other* metric + sym_base + covariates fixed at session-specific means/mode
      - global x-axis per metric across both sessions
    """
    x1, x2 = metrics
    dfp = df_clean.copy()
    dfp["session"] = dfp["session"].astype(int)

    # Ensure required columns exist
    required = {x1, x2, "sym_next", "sym_base", "age", "gender", "session"}
    missing = required - set(dfp.columns)
    if missing:
        raise ValueError(f"df_clean is missing columns required for plotting: {sorted(missing)}")

    # global x-axis limits per metric (same across both sessions)
    xlims = {
        x1: (float(dfp[x1].min()), float(dfp[x1].max())),
        x2: (float(dfp[x2].min()), float(dfp[x2].max())),
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        constrained_layout=True,
        sharex="col"
    )

    # Columns
    panels = [
        (0, x1, x2, "Δ cycle rate (pre − post)"),
        (1, x2, x1, "Δ cycle strength (pre − post)"),
    ]

    def plot_panel(ax, dsub, model, x_term, other_term, title=None):
        # raw scatter: observed next-session symptoms
        ax.scatter(
            dsub[x_term].astype(float),
            dsub["sym_next"].astype(float),
            alpha=0.5,
            s=18,
            color="black"
        )

        # prediction grid (global x for that metric)
        x_min, x_max = xlims[x_term]
        x_vals = np.linspace(x_min, x_max, n_grid)

        # fixed predictors for marginal line (session-specific means/mode)
        pred_df = pd.DataFrame({
            x_term: x_vals,
            other_term: float(dsub[other_term].astype(float).mean()),
            "sym_base": float(dsub["sym_base"].astype(float).mean()),
            "age": float(dsub["age"].astype(float).mean()),
            "gender": dsub["gender"].mode().iloc[0],  # formula uses C(gender)
        })

        pred = model.get_prediction(pred_df)
        mean = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)

        ax.plot(x_vals, mean, color="black", linewidth=3)
        ax.fill_between(x_vals, ci[:, 0], ci[:, 1], color="black", alpha=0.2)

        ax.set_xlim(x_min, x_max)

        if title is not None:
            ax.set_title(title)

        ax.set_xlabel("")  # set below per column
        ax.set_ylabel(y_label)

    # draw panels
    for row, sess in enumerate([1, 2]):
        if sess not in models:
            raise ValueError(f"models is missing key {sess}. Expected models={{1:..., 2:...}}")

        dsub = dfp[dfp["session"] == sess].copy()
        m = models[sess]

        for col, x_term, other_term, col_title in panels:
            ax = axes[row, col]

            title = col_title if row == 0 else None
            plot_panel(ax, dsub, m, x_term, other_term, title=title)

            ax.text(
                0.02, 0.98,
                f"Transition {sess} ({sess}→{sess+1})",
                transform=ax.transAxes,
                va="top", ha="left", color="black"
            )

            # x-label on bottom row only
            if row == 1:
                ax.set_xlabel(col_title)

            # remove duplicate y-labels on right column
            if col == 1:
                ax.set_ylabel("")

    # final styling (match your FO plot)
    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    if savepath:
        fig.savefig(savepath + ".png", dpi=300, bbox_inches="tight")
        fig.savefig(savepath + ".svg", bbox_inches="tight")

    return fig, axes


def plot_cycle_params_baseline_hads(df, ses_idx, n_states_level1, covariates=('age', 'gender'), robust='HC3'):
    """
    Plot cycle dynamics.
    
    Parameters:
    - df: dataframe with cycle features and demographics
    - ses_idx: which session (99 is all sessions concatenated)
    - n_states_level1: how many states
    -covariates: covariates for partial regression plot
    """

    # brain state colours for cycle
    brain_state_colors = sns.color_palette("tab20", n_colors=n_states_level1)  
    rgb = np.array(brain_state_colors)        # shape: (n_states_level1, 3)

    # set up figure
    fig = plt.figure(figsize=(7.1, 8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.4)    

    # Panel A: FO Asymmetry Matrix
    asym, significant = calc_sign_asymmetry(output_dir, ses_idx, n_states_level1)

    # Plot mean asymmetry (mean over sessions and patients)
    mean_asym = asym.mean(axis=-1) 

    # 1 to 10
    state_labels = [str(i) for i in range(1, n_states_level1 + 1)]
    ax1 = fig.add_subplot(gs[0, 0])
    ax1 = sns.heatmap(mean_asym, cmap='viridis', center=0, xticklabels=state_labels,
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

    ax1.text(-0.15, 1.5, "a", transform=ax1.transAxes,
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
        plt.cm.ScalarMappable(norm=norm, cmap='viridis'),
        cax=cax,
        orientation='horizontal'
    )

    # Only show extremes
    cbar.set_ticks([-.07, 0, .07])

    # Format ticks to 2 decimals
    cbar.ax.yaxis.set_major_formatter(
        mticker.FormatStrFormatter('%.2f')
    )
    cbar.ax.set_title('FO asymmetry', pad=8)

    # Panel B: cycle

    # load tinda
    with open(f"{output_dir}/tinda_{sess_idx}_{n_states_level1}.pkl", "rb") as f:
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

    ax2.text(-0.15, 1.5, "b", transform=ax2.transAxes,
        fontsize=20, fontweight="bold", va="top")

    # baseline regressions

    # drop missing essentials
    dfp = df.dropna(subset=["hads_dep_total", "cycle_strength", "cycle_rate", "age", "gender"]).copy()

    # optional outlier removal (keep consistent with your other figures)
    dfp = dfp[
        (np.abs(zscore(dfp["cycle_strength"].astype(float), nan_policy="omit")) < 3) &
        (np.abs(zscore(dfp["cycle_rate"].astype(float), nan_policy="omit")) < 3) &
        (np.abs(zscore(dfp["hads_dep_total"].astype(float), nan_policy="omit")) < 3)
    ].copy()

    m_strength = smf.ols("cycle_strength ~ hads_dep_total + age + C(gender)", data=dfp).fit(cov_type=robust)
    m_rate     = smf.ols("cycle_rate ~ hads_dep_total + age + C(gender)", data=dfp).fit(cov_type=robust)

    cov_pred = {
        "age": float(dfp["age"].mean()),
        "gender": dfp["gender"].mode().iloc[0],
    }

    ax3 = fig.add_subplot(gs[1, 0])
    plot_reg_prediction_style(
        ax=ax3,
        model=m_strength,
        data=dfp,
        x_col="hads_dep_total",
        y_col="cycle_strength",
        covariate_cols_for_pred=cov_pred,
        xlabel="baseline HADS-D",
        ylabel="baseline cycle strength",
        panel_letter="c",
    )

    ax4 = fig.add_subplot(gs[1, 1])
    plot_reg_prediction_style(
        ax=ax4,
        model=m_rate,
        data=dfp,
        x_col="hads_dep_total",
        y_col="cycle_rate",
        covariate_cols_for_pred=cov_pred,
        xlabel="baseline HADS-D",
        ylabel="baseline cycle rate",
        panel_letter="d",
    )

    # Remove top/right spines for all axes (you already do this; harmless to keep)
    for ax in fig.axes:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.savefig(f"{output_dir}/baseline_cycle_params_{n_states_level1}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{output_dir}/baseline_cycle_params_{n_states_level1}.svg", dpi=300, bbox_inches="tight")
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

    if ax is None:
        fig, ax = plt.subplots(figsize=(4, 4))
    else:
        fig = ax.figure

    # make sure colors match first level HMM states
    if color_scheme is None:
        color_scheme = sns.color_palette("tab20", n_colors=20)

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


    theta = np.linspace(0, -2 * np.pi, K, endpoint=False)
    theta += np.pi / 2  # start at 12 o'clock

    radius = 1
    coords = radius * np.column_stack((np.cos(theta), np.sin(theta)))


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

    ax.set_aspect("equal")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.axis("off")

    return fig, ax


def calc_sign_asymmetry(output_dir, ses_idx, n_states_level1):

    # load asymmetry matrix
    asym = np.load(f'{output_dir}/fo_asymmetry_{ses_idx}_{n_states_level1}.npy')

    # Prepare t-test outputs
    t_vals = np.zeros((n_states_level1, n_states_level1))
    p_vals = np.ones((n_states_level1, n_states_level1))

    # List for Bonferroni count
    tests = []

    # Run t-tests
    for i in range(n_states_level1):
        for j in range(n_states_level1):
            if i == j:
                continue

            vals = asym[i, j, :]

            t, p = ttest_1samp(vals, 0, nan_policy="omit")
            t_vals[i, j] = t
            p_vals[i, j] = p
            tests.append((i, j, p))

    # Bonferroni correction (as in van Es, Higgins et al.)
    n_tests = len(tests)
    alpha_corr = 0.05 / n_tests
    significant = p_vals < alpha_corr

    return asym, significant


def plot_reg_prediction_style(
    ax,
    model,
    data,
    x_col,
    y_col,
    covariate_cols_for_pred,   # dict of fixed values
    xlabel,
    ylabel,
    panel_letter=None,
    n_grid=200,
):
    ax.scatter(data[x_col], data[y_col], alpha=0.5, s=18, color="black")

    x_vals = np.linspace(float(data[x_col].min()), float(data[x_col].max()), n_grid)

    pred_df = pd.DataFrame({x_col: x_vals})
    for k, v in covariate_cols_for_pred.items():
        pred_df[k] = v

    pred = model.get_prediction(pred_df)
    mean = pred.predicted_mean
    ci = pred.conf_int(alpha=0.05)

    ax.plot(x_vals, mean, color="black", linewidth=3)
    ax.fill_between(x_vals, ci[:, 0], ci[:, 1], color="black", alpha=0.2)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    if panel_letter is not None:
        ax.text(
            -0.15, 1.15, panel_letter,
            transform=ax.transAxes,
            fontsize=20,
            fontweight="bold",
            va="top"
        )


def load_cycle_parameters(n_states_level1: int, n_states_level2: int, sess_idx: int):

    with open(output_dir / f"tinda_{sess_idx}_{n_states_level1}.pkl", "rb") as f:
        tinda = pickle.load(f)

    cycle_strength = tinda["cycle_strength"]

    fp = hmm_dir / "figures" / "cycles" / f"cycle_rate_{n_states_level1}_{n_states_level2}" / "run1" / "cycle_duration.pkl"
    with open(fp, "rb") as f:
        cycle_duration = pickle.load(f)

    return cycle_strength, cycle_duration


def add_cycle_parameters_to_df(n_states_level1: int, sess_idx: int):

    # load the clinical data including HMM parameters
    csv_path = Path(f"{hmm_dir}/hmm_demo_hads2704_{n_states_level1}.csv")

    df = pd.read_csv(csv_path)

    print(f"Analyzing {df['patient'].nunique()} patients") 

    # states should be 1 to 10
    df["state"] = df["state"] + 1

    # a bit of a clean up
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    for col in ["patient", "session", "tms", "state", "group", "responder", "gender"]:
        df[col] = df[col].astype("category", errors="ignore")

    # select only state 2 (anterior DMN state)
    df_state1 = df[df["state"] == 2].copy()

    cycle_strength, cycle_duration = load_cycle_parameters(n_states_level1, n_states_level1, sess_idx)

    # @ Mats: do you average over cycle durations within individuals?
    cycle_mean = [np.mean(c) for c in cycle_duration]
    df_state1["cycle_strength"] = cycle_strength
    df_state1["cycle_duration"] = cycle_mean
    df_state1["cycle_rate"] = 1.0 / df_state1["cycle_duration"].astype(float)

    out_csv = hmm_dir / f"df_includingcycles2704_{n_states_level1}.csv"
    df_state1.to_csv(out_csv, index=False)

    return df_state1


def load_and_prep_data(n_states_level1, exclude_repeater: bool = False):
    
    # read df including cycle params
    csv_path = Path(f'{hmm_dir}/df_includingcycles2704_{n_states_level1}.csv')

    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    unique_ids = pd.unique(pd.Series(df['patient']))
    
    df = add_recording_length(df)

    # exclude patients that repeated TMS treatment?
    if exclude_repeater:
        repeater_ids = [i for i in unique_ids if "R" in str(i)]
        print(f'{len(repeater_ids)} patients repeated the treatment')
        repeater_positions = [list(unique_ids).index(i) for i in repeater_ids]
        df = df[~df["patient"].str.contains("R")]
        drop_indices = repeater_positions
        np.save(f'{hmm_dir}/dropped_indices.npy', np.array(drop_indices))

    print(f"Analyzing {df['patient'].nunique()} patients")

    # transform to categorical
    for col in ["patient", "session", "tms", "state", 'responder', 'group', 'gender']:
        df[col] = df[col].astype("category")

    return df


def add_recording_length(df):
    # merge EEG recording length
    eeg_rec_length = pd.read_csv(eeg_length_path)

    df["patient"] = df["patient"].astype(str)
    eeg_rec_length["patient"] = eeg_rec_length["patient"].astype(str)

    df["session"] = pd.to_numeric(df["session"], errors="coerce").astype("Int64")
    eeg_rec_length["session"] = pd.to_numeric(eeg_rec_length["session"], errors="coerce").astype("Int64")
    df["tms"] = df["tms"].astype(str).str.lower()

    # map EEG recording index (0..5) to clinical session (1..3) and tms (pre/post)
    eeg_rec_length["session_clinical"] = (eeg_rec_length["session"] // 2) + 1
    eeg_rec_length["tms"] = np.where(eeg_rec_length["session"] % 2 == 0, "pre", "post")

    # keep only relevant columns
    eeg_rec_length = eeg_rec_length[["patient", "session_clinical", "tms", "eeg_recording_length"]]
    eeg_rec_length = eeg_rec_length.drop_duplicates(subset=["patient", "session_clinical", "tms"])

    # rename for merge
    eeg_rec_length = eeg_rec_length.rename(columns={"session_clinical": "session"})
    
    # merge on patient + session + tms
    df = df.merge(
        eeg_rec_length,
        on=["patient", "session", "tms"],
        how="left"
    )

    df["eeg_recording_length"] = (
    df["eeg_recording_length"] - df["eeg_recording_length"].mean()
        ) / df["eeg_recording_length"].std()

    return df


if __name__ == 'main':
    for n_states in n_states_level1:
        #print(f"{n_states}")
        # run this once to get a dataframe with cycle parameters added to clinical data
        df_state2 = add_cycle_parameters_to_df(n_states, 99)
        # baseline analysis only
        analyse_cycle_params(n_states)
        # baseline and symptom change
        plot_pc_vs_symptom_change(n_states)
