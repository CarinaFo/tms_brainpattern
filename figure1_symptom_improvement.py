# Plot figure 1 (symptom improvement in HADS score over 3 sessions)
# run descriptives and stats on baseline predictors for results
# run in python 3.12 environment

import os
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns

import statsmodels.formula.api as smf

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

# --------------------------------------------------
# Configuration
# --------------------------------------------------
home_dir = Path("L:/Lab_LucaC/Carina/")
csv_path = Path(f"{home_dir}/canonical_hmm_finalsample/clinical_demo_combined_012026.csv")
fig_dir = Path(f'{home_dir}/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered/figures')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

#plot figure 1 (symptom change, coefficients)
#plot_symptom_change_hads()
#plot_regression_coeffs_demo()

# Colorblind-friendly palette (Okabe-Ito)
cb_palette = ["#E69F00", "#56B4E9", "#009E73", "#F0E442", "#0072B2",
              "#D55E00", "#CC79A7", "#999999", "#000000"]

base_color = cb_palette[0]

def lighten_color(color, amount=0.6):
    r, g, b = mcolors.to_rgb(color)
    return (
        r + (1 - r) * amount,
        g + (1 - g) * amount,
        b + (1 - b) * amount,
    )

light_color = lighten_color(base_color, amount=0.6)

# --------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------
def load_and_prep_data(exclude_repeater: bool = False):

    df = pd.read_csv(csv_path)

    # ensure patient IDs are strings
    df["patient"] = df["patient"].astype(str)

    if exclude_repeater:
        df = df[~df["patient"].str.contains("R", na=False)]
    
    # load patient ID list
    patient_ids = pd.read_csv(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"))

    patient_ids["patient_id"] = patient_ids["patient_id"].astype(str)

    # keep only patients fitted for this HMM
    df = df[df["patient"].isin(patient_ids["patient_id"])]

    print(f"Analyzing {df['patient'].nunique()} patients")

    # compute years with depression
    df["years_with_depression"] = df["age"] - df["age of symptom onset"]

    return df


def plot_symptom_change_hads(all_weeks: bool = False, outcome_variable: str = 'hads_dep_total'):

    df = load_and_prep_data()

    session_order = [
    "pre",
    "week 1",
    "week 2",
    "week 3",
    "week 4",
    "week 5",
    "week 6",
    "post",
    ]

    # Create dictionary mapping session → integer
    session_map = {sess: i for i, sess in enumerate(session_order)}

    df_mid = df[df["session"].isin(session_order)].copy()

    df_mid["session_num"] = df_mid["session"].map(session_map)

    lmm_linear = smf.mixedlm(
        f"{outcome_variable} ~ session_num",
        data=df_mid,
        groups=df_mid["patient"]
        ).fit(reml=False)

    print(lmm_linear.summary())

    lmm_quad = smf.mixedlm(
        f"{outcome_variable} ~ session_num + I(session_num**2)",
        data=df_mid,
        groups=df_mid["patient"]
    ).fit(reml=False)

    print(lmm_quad.summary())

    lr_stat = 2 * (lmm_quad.llf - lmm_linear.llf)
    df_diff = lmm_quad.df_modelwc - lmm_linear.df_modelwc

    from scipy.stats import chi2
    p_value = chi2.sf(lr_stat, df_diff)

    print(f"LR χ²({df_diff}) = {lr_stat:.2f}, p = {p_value:.4f}")

    # Group by patient and session, then get the dep_hads score
    var_per_session = df.groupby(['patient', 'session'])[outcome_variable].first().reset_index()
    
    var_per_session = var_per_session.dropna(
    subset=[outcome_variable]
    )

    if all_weeks:
        session_order = [
        "pre",
        'week 1',
        'week 2',
        "week 3",
        'week 4',
        'week 5',
        'week 6',
        "post",
        ]

        # Keep only desired sessions
        var_per_session = var_per_session[
            var_per_session["session"].isin(session_order)
        ]

        # Make session an ordered categorical variable
        var_per_session["session"] = pd.Categorical(
        var_per_session["session"],
        categories=session_order,
        ordered=True
        )

        # for comparision with Marino et al., preprint
        sns.lineplot(
            x="session",
            y=outcome_variable,
            data=var_per_session,
            estimator="mean",
            errorbar="ci",
            marker="o",
            linewidth=4
        )

        plt.ylabel(f"HADS-D")
        plt.xlabel("Treatment timepoint")
        sns.despine()

        plt.tight_layout()
        plt.savefig(f"{fig_dir}/allweeks_dep_hads.svg", dpi=300, bbox_inches="tight")
        plt.savefig(f"{fig_dir}/allweeks_dep_hads.png", dpi=300, bbox_inches="tight")
        plt.show()


    session_order = [
        "pre",
        "week 3",
        "post",
        ]

    # Keep only desired sessions
    var_per_session = var_per_session[
        var_per_session["session"].isin(session_order)
    ]

    # -------------------------------
    # KEEP ONLY COMPLETE PATIENTS
    # -------------------------------
    required_sessions = set(session_order)

    patients_with_all_sessions = (
        var_per_session
        .groupby("patient")["session"]
        .apply(set)
        .loc[lambda s: s == required_sessions]
        .index
    )

    var_per_session = var_per_session[
        var_per_session["patient"].isin(patients_with_all_sessions)
    ]

    # Make session an ordered categorical variable
    var_per_session["session"] = pd.Categorical(
        var_per_session["session"],
        categories=session_order,
        ordered=True
    )

    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(7.1, 6))
    ax = plt.gca()

    # --- Full violin ---
    sns.violinplot(
        data=var_per_session,
        x="session",
        y=outcome_variable,
        color=base_color,
        inner=None,        # remove default box/median
        cut=0,
        alpha=0.5,
        linewidth=0,
        ax=ax
    )

    # --- Convert to HALF violin (left side only) ---
    for i, artist in enumerate(ax.collections):
        path = artist.get_paths()[0]
        vertices = path.vertices
        mean_x = vertices[:, 0].mean()
        vertices[:, 0] = np.minimum(vertices[:, 0], mean_x)  # keep left half

    # --- Single-subject trajectories (light orange) ---
    session_order = var_per_session["session"].cat.categories
    x_positions = {s: i for i, s in enumerate(session_order)}

    for patient, df_p in var_per_session.groupby("patient"):
        df_p = df_p.sort_values("session")
        ax.plot(
            df_p["session"].map(x_positions),
            df_p[outcome_variable],
            color="orange",
            alpha=0.1,
            linewidth=1.5,
            zorder=1
        )

    # --- Median markers and IQR ---
    mean_vals = []  # to store median for each session
    for i, session in enumerate(session_order):
        vals = var_per_session[var_per_session["session"] == session][outcome_variable]
        mean = np.mean(vals)
        mean_vals.append(mean)

    # --- Dark-orange line connecting medians ---
    ax.plot(
        range(len(session_order)),
        mean_vals,
        color="orange",
        linewidth=3,
        marker='D',
        zorder=4
    )

    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Treatment time", labelpad=10)
    ax.set_ylabel("HADS-D", labelpad=10)

    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([0, 10, 20])
    ax.set_ylim(-1, 22)

    # Correct x-ticks
    ax.set_xticks(range(len(session_order)))
    ax.set_xticklabels(["Pre", "Mid", "Post"])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sns.despine()

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/raincloud_dep_hads_trajectories.svg", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_dir}/raincloud_dep_hads_trajectories.png", dpi=300, bbox_inches="tight")
    plt.show()

    descriptives(var_per_session, outcome_variable, df)


def descriptives(var_per_session, outcome_variable, df):

    wide = var_per_session.pivot(
    index="patient",
    columns="session",
    values=outcome_variable
    )

    wide["percent_improvement"] = (
    (wide["pre"] - wide["post"]) / wide["pre"]
    )

    wide["absolute_improvement"] = (
    (wide["pre"] - wide["post"])
    )

    wide["abs_pre_mid"] = (
    (wide["pre"] - wide["week 3"])
    )

    wide["abs_mid_post"] = (
    (wide["week 3"] - wide["post"])
    )

    responders = wide["percent_improvement"] >= 0.5

    n_responders = responders.sum()
    n_total = wide.shape[0]

    print(f"{n_responders} / {n_total} patients improved ≥50% from baseline to post")

    df.groupby('patient')['gender'].first().value_counts()
    df.groupby('patient')['age'].first().median()
    df.groupby('patient')['age'].first().min()
    df.groupby('patient')['age'].first().max()

    df.groupby('patient')['years_with_depression'].first().mean()
    df.groupby('patient')['years_with_depression'].first().min()
    df.groupby('patient')['years_with_depression'].first().max()

    df.groupby('patient')['years_with_depression'].first().mean()
    df.groupby('patient')['years_with_depression'].first().min()
    df.groupby('patient')['years_with_depression'].first().max()

    def extract_leading_number(s):
        return (
        s.astype(str)
         .str.extract(r'(\d+)', expand=False)
         .astype(float)
    )

    df['treatment_days_num'] = extract_leading_number(df['number of treatment days'])

    df.groupby('patient')['treatment_days_num'].first().mean()
    df.groupby('patient')['treatment_days_num'].first().min()

    baseline_vars = (
    df[df["session"] == "pre"]
    .set_index("patient")[
        [
            "age",
            "gender",
            "years_with_depression",
            "age of symptom onset",
            "previous ect",
            "previous tms",
            "research tier",
            'treatment_days_num'
        ]
    ]
    .rename(
        columns={
            "age of symptom onset": "age_onset",
            "previous ect": "previous_ect",
            "previous tms": "previous_tms",
            "research tier": "research_tier"
            }
        )
    )
    
    # clean up treatment days
    baseline_vars.groupby('patient')['treatment_days_num'].first().mean()

    model_df = wide.join(baseline_vars, how="inner")

    categorical_vars = [
    "gender",
    "previous_ect",
    "previous_tms",
    "research_tier",
    ]

    for col in categorical_vars:
        model_df[col] = model_df[col].astype("category")

    # standardize numerical variables
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    continuous_vars = ["age", "age_onset", "treatment_days_num", "pre"]

    model_df[continuous_vars] = scaler.fit_transform(model_df[continuous_vars])

    base_model = smf.ols('post ~  pre + age + gender + age_onset + previous_ect + previous_tms + research_tier + treatment_days_num'
                    , data=model_df).fit()
    
    print(base_model.summary())

    # plot the coefficients
    coefs = base_model.params
    conf = base_model.conf_int()
    conf.columns = ['lower', 'upper']

    coefs_df = conf.copy()
    coefs_df['coef'] = coefs

    # Drop intercept
    coefs_df = coefs_df.drop('Intercept').reset_index().rename(columns={'index':'Predictor'})

    # Create a mapping from model variable names → friendly labels
    rename_dict = {
        "age": "age",
        "years_with_depression": "Years with depression",
        "gender[T.Male]": "Gender: Male vs Female",
        "research_tier[T.Naturalistic]": "Naturalistic vs RCT",
        "research_tier[T.RCT acceptable]": "Neurological vs RCT",
        "pre": "Baseline hads score",
        "age_onset": "Depression onset",
        "treatment_days_num": "Treatment days",
        "previous_ect[T.Yes]": "Previous ECT",
        "previous_tms[T.Yes]": "Previous TMS",
    }

    coefs_df["Predictor"] = coefs_df["Predictor"].replace(rename_dict)

    plt.figure(figsize=(8,4))
    ax = sns.pointplot(
        data=coefs_df,
        x='coef',
        y='Predictor',
        join=False,
        color='black'
    )

    # Add 95% confidence intervals manually
    for i, row in coefs_df.iterrows():
        plt.plot([row['lower'], row['upper']], [i, i], color='black', lw=2)

    # Add reference line at zero
    plt.axvline(0, color='gray', linestyle='--', lw=1)
    # Set y-axis tick labels fontsize to 12
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=18)
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=18)

    # Labels and aesthetics
    plt.xlabel("Regression Coefficient (β)", fontsize=18)
    plt.ylabel("")
    ax.xaxis.grid(True, linestyle='--', alpha=0.6)
    ax.yaxis.grid(False)
    sns.despine()
    plt.tight_layout()

    # -------------------------
    # Save figure
    # -------------------------
    plt.savefig(Path(f'{fig_dir}/coefplot_hads_predictors.png'), dpi=300, bbox_inches='tight')
    plt.savefig(Path(f'{fig_dir}/coefplot_hads_predictors.svg'), dpi=300, bbox_inches='tight')
    plt.show()


def plot_change_correlations_quadrants(
    df: pd.DataFrame,
    patient_col: str = "patient",
    session_col: str = "session",
    scales: list[str] = ['hads_dep_total', 'hads_anx_total', 'hama_total', 'madrs_total'],
    session_a: int = 'pre',
    session_b: int = 'post',
    method: str = "spearman",
    dropna_rows: bool = True,
    figsize_per_cell: float = 3.2,
):
    """
    Compute within-patient change (session_b - session_a) for each scale, then
    plot pairwise scatter plots with 4 quadrants and display Spearman rho.

    Returns:
        diff (pd.DataFrame): patient-level change scores (index = patient)
        corr (pd.DataFrame): correlation matrix across change scores
    """
    if scales is None:
        raise ValueError("Please provide a list of scale column names in `scales`.")

    # --- compute change scores (wide -> subtract) ---
    wide = df.pivot_table(index=patient_col, columns=session_col, values=scales, aggfunc="mean")

    # Ensure required sessions exist as columns; if not, you'll get KeyError
    try:
        a = wide.xs(session_a, level=session_col, axis=1)
        b = wide.xs(session_b, level=session_col, axis=1)
    except KeyError as e:
        raise KeyError(
            f"Could not find sessions {session_a} and/or {session_b} in '{session_col}'. "
            f"Available sessions: {sorted(df[session_col].dropna().unique())}"
        ) from e

    # pre minus post to measure improvement
    diff = (a-b).copy()
    diff.columns = [f"{c}_diff_s{session_a}_s{session_b}" for c in diff.columns]

    if dropna_rows:
        diff = diff.dropna(how="any")

    # --- correlation matrix ---
    corr = diff.corr(method=method)

    # --- plot pairwise grid (lower triangle) ---
    n = len(scales)
    fig, axes = plt.subplots(n, n, figsize=(figsize_per_cell * n, figsize_per_cell * n))

    # Helper to map original scale -> diff column
    diff_col = {s: f"{s}_diff_s{session_a}_s{session_b}" for s in scales}

    for i, y_scale in enumerate(scales):
        for j, x_scale in enumerate(scales):
            ax = axes[i, j]

            if i == j:
                # Diagonal: show label + N
                ax.axis("off")
                ax.text(
                    0.5, 0.6, y_scale,
                    ha="center", va="center", fontweight="bold",
                    transform=ax.transAxes
                )
                ax.text(
                    0.5, 0.4, f"N={len(diff)}",
                    ha="center", va="center", 
                    transform=ax.transAxes
                )
                continue

            x = diff[diff_col[x_scale]]
            y = diff[diff_col[y_scale]]

            # Hide upper triangle to reduce clutter
            if j > i:
                ax.axis("off")
                continue

            # Scatter
            ax.scatter(x, y)

            # Quadrant lines
            ax.axhline(0)
            ax.axvline(0)

            # Spearman rho (pairwise complete)
            mask = x.notna() & y.notna()
            if mask.sum() >= 3:
                rho = x[mask].corr(y[mask], method=method)
                ax.text(
                    0.05, 0.95, f"{method.title()} ρ={rho:.2f}",
                    ha="left", va="top", transform=ax.transAxes
                )
            else:
                ax.text(
                    0.05, 0.95, "Too few pairs",
                    ha="left", va="top", transform=ax.transAxes
                )

            # Labels on left column / bottom row only
            if j == 0:
                ax.set_ylabel(f"{y_scale}\nΔ(S{session_a}-S{session_b})")
            else:
                ax.set_yticklabels([])

            if i == n - 1:
                ax.set_xlabel(f"{x_scale}\nΔ(S{session_a}-S{session_b})")
            else:
                ax.set_xticklabels([])

    fig.tight_layout()
    plt.savefig(f'{fig_dir}/clinical_scores_corr.svg')
    plt.show()

    return diff, corr
