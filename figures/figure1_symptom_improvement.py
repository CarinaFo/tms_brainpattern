# Plot figure 1 (symptom improvement in HADS score over 3 sessions)
# run descriptives and stats on baseline predictors for results
import os
from pathlib import Path
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
import ptitprince as pt

import statsmodels.formula.api as smf

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42


plt.rcParams.update({
    "font.family": "Arial",  # Nature preference
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# --------------------------------------------------
# Configuration
# --------------------------------------------------
home_dir = Path("L:/Lab_LucaC/Carina/")
csv_path = Path(f"{home_dir}/canonical_hmm_finalsample/clinical_demo_combined_012026.csv")
fig_dir = Path(f'{home_dir}/canonical_hmm_finalsample/figures')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# plot figure 1 (symptom change, coefficients)
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
def load_and_prep_data(csv_path, exclude_repeater: bool = False):

    df = pd.read_csv(csv_path)

    # ensure patient IDs are strings
    df["patient"] = df["patient"].astype(str)

    if exclude_repeater:
        df = df[~df["patient"].str.contains("R", na=False)]
    
    # 159 is missing post HADS
    df = df[~df["patient"].str.contains("159", na=False)]
        
    # load patient ID list
    patient_ids = pd.read_csv(Path("L:/Lab_LucaC/Carina/canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"))

    patient_ids["patient_id"] = patient_ids["patient_id"].astype(str)

    # keep only patients fitted for this HMM
    df = df[df["patient"].isin(patient_ids["patient_id"])]

    print(f"Analyzing {df['patient'].nunique()} patients")

    # compute years with depression
    df["years_with_depression"] = df["age"] - df["age of symptom onset"]

    return df


def plot_symptom_change_hads(all_weeks: bool, outcome_variable: str):

    df = load_and_prep_data(csv_path)

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
        "hads_dep_total ~ session_num",
        data=df_mid,
        groups=df_mid["patient"]
        ).fit(reml=False)

    print(lmm_linear.summary())

    lmm_quad = smf.mixedlm(
        "hads_dep_total ~ session_num + I(session_num**2)",
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
            errorbar="se",
            marker="o",
            linewidth=2
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

    # Plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial'

    plt.figure(figsize=(7.1, 6))

    ax = pt.RainCloud(
        x="session",
        y=outcome_variable,
        data=var_per_session,
        color=base_color,
        bw=.25,
        width_viol=.6,
        move=.25,
        alpha=0.9,
        orient="v",
        point_size=0
    )

    # --- Individual trajectories ---
    session_order = var_per_session["session"].cat.categories
    x_positions = {s: i for i, s in enumerate(session_order)}

    for patient, df_p in var_per_session.groupby("patient"):
        df_p = df_p.sort_values("session")
        ax.plot(
            df_p["session"].map(x_positions),
            df_p[outcome_variable],
            color="grey",
            alpha=0.2,
            linewidth=1,
            zorder=0
        )

    # Styling
    ax.yaxis.grid(True, linestyle="--", linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Treatment timepoint", labelpad=10)
    ax.set_ylabel("HADS-D", labelpad=10)

    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([0, 10, 20])
    ax.set_ylim(-1, 22)

    ax.set_xticklabels(
        [s.replace("week ", "Week ").title() for s in session_order]
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sns.despine()

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/raincloud_dep_hads_trajectories.svg", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_dir}/raincloud_dep_hads_trajectories.png", dpi=300, bbox_inches="tight")
    plt.show()

    descriptives()


def descriptives(df: pd.DataFrame, outcome_variable: str, var_per_session: pd.DataFrame):

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
    df.groupby('patient')['age'].first().mean()
    df.groupby('patient')['age'].first().min()
    df.groupby('patient')['age'].first().max()

    df.groupby('patient')['years_with_depression'].first().mean()
    df.groupby('patient')['years_with_depression'].first().min()
    df.groupby('patient')['years_with_depression'].first().max()

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
            'number of treatment days'
        ]
    ]
    .rename(
        columns={
            "age of symptom onset": "age_onset",
            "previous ect": "previous_ect",
            "previous tms": "previous_tms",
            "research tier": "research_tier",
            "number of treatment days": 'number_of_treatment_days'
            }
        )
    )
    
    # clean up treatment days
    baseline_vars["number_of_treatment_days"] = (baseline_vars["number_of_treatment_days"].astype(str).str.extract(r'(\d+)')).astype('int')
    
    baseline_vars.groupby('patient')['number_of_treatment_days'].first().mean()

    model_df = wide.join(baseline_vars, how="inner")

    categorical_vars = [
    "gender",
    "previous_ect",
    "previous_tms",
    #"research_tier",
    ]

    for col in categorical_vars:
        model_df[col] = model_df[col].astype("category")

    # standardize numerical variables
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    continuous_vars = ["age", "age_onset", "number_of_treatment_days", "pre"]

    model_df[continuous_vars] = scaler.fit_transform(model_df[continuous_vars])

    base_model = smf.ols('post ~  pre + age + gender + age_onset + previous_ect + previous_tms + research_tier + number_of_treatment_days'
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
        "number_of_treatment_days": "Treatment days",
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