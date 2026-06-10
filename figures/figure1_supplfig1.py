# Figure 1: Symptom improvement + supplementary figure 1: Symptoms over 4 weeks, baseline predictors,
# correlation HADS-D * MADRS
from pathlib import Path
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import seaborn as sns
from scipy.stats import spearmanr
import statsmodels.formula.api as smf


# text editable in inkscape
plt.rcParams['svg.fonttype'] = 'none'

plt.rcParams['pdf.fonttype'] = 42
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
})

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

home_dir = Path("L:/Lab_LucaC/Carina/")
csv_path = home_dir / "canonical_hmm_finalsample/clinical_demo_combined_270426.csv"
fig_dir_suppl = home_dir / "canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered/figures/supplementary_figures"
fig_dir_suppl.mkdir(parents=True, exist_ok=True)

fig_dir_main = home_dir / "canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered/figures/main_figures"
fig_dir_main.mkdir(parents=True, exist_ok=True)


def load_and_prep_data(exclude_repeater=False):
    df = pd.read_csv(csv_path)
    df["patient"] = df["patient"].astype(str)

    if exclude_repeater:
        df = df[~df["patient"].str.contains("R", na=False)]

    patient_ids = pd.read_csv(
        home_dir / "canonical_hmm_finalsample/prepared_data_giles_1Hz_3Hzfiltereddata/patients_fitted_for_this_hmm.csv"
    )
    patient_ids["patient_id"] = patient_ids["patient_id"].astype(str)
    df = df[df["patient"].isin(patient_ids["patient_id"])]
    
    df['age_onset'] = pd.to_numeric(df['age of symptom onset'], errors='coerce')

    df["years_with_depression"] = df["age"] - df["age_onset"]

    df["treatment_days_num"] = (
        df["number of treatment days"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .astype(float)
    )
    return df

def plot_symptoms_pre_w2_w4(outcome_variable="hads_dep_total"):

    df = load_and_prep_data()

    session_order = ["pre", "week 2", "week 4"]

    plot_df = (
        df.groupby(["patient", "session"])[outcome_variable]
        .first()
        .reset_index()
        .dropna(subset=[outcome_variable])
    )
    plot_df = plot_df[plot_df["session"].isin(session_order)].copy()

    plot_df["session"] = pd.Categorical(
        plot_df["session"],
        categories=["pre", "week 2", "week 4"],
        ordered=True
    )

    model = smf.mixedlm(
        f"{outcome_variable} ~ session",
        data=plot_df,
        groups=plot_df["patient"]
    ).fit()

    print(model.summary())

    plot_df["session"] = pd.Categorical(
        plot_df["session"], categories=session_order, ordered=True
    )

    fig, ax = plt.subplots(figsize=(4, 3.6))

    # -------------------------
    # Half violin
    # -------------------------
    sns.violinplot(
        data=plot_df,
        x="session",
        y=outcome_variable,
        color=base_color,
        inner=None,
        cut=0,
        linewidth=0,
        ax=ax,
    )

    # convert full violins to left half only
    for artist in ax.collections:
        try:
            path = artist.get_paths()[0]
            vertices = path.vertices
            mean_x = vertices[:, 0].mean()
            vertices[:, 0] = np.minimum(vertices[:, 0], mean_x)
        except Exception:
            pass
    # -------------------------
    # Individual trajectories
    # -------------------------
    x_positions = {s: i for i, s in enumerate(session_order)}
    for _, df_p in plot_df.groupby("patient"):
        df_p = df_p.sort_values("session")
        ax.plot(
            df_p["session"].map(x_positions),
            df_p[outcome_variable],
            color="orange",
            alpha=0.15,
            linewidth=1,
            zorder=1
        )

    # -------------------------
    # Mean line
    # -------------------------
    mean_vals = []
    for session in session_order:
        vals = plot_df.loc[plot_df["session"] == session, outcome_variable]
        mean_vals.append(np.mean(vals))

    ax.plot(
        range(len(session_order)),
        mean_vals,
        color="orange",
        linewidth=2.5,
        marker="D",
        zorder=4
    )

    ax.set_xlabel("")
    ax.set_ylabel("HADS-D", fontsize=11)
    ax.set_xticklabels(["Session 1", "Session 11", "Session 20"], fontsize=11)

    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([0, 10, 20], fontsize=11)
    ax.set_ylim(-1, 22)

    ax.text(
        0.0, 1.02, "b",
        transform=ax.transAxes,
        fontsize=20,
        ha="left",
        va="bottom",
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    sns.despine()

    plt.tight_layout()
    plt.savefig(f"{fig_dir_main}/symptoms_pre_week2_week4.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_dir_main}/symptoms_pre_week2_week4.svg", dpi=300, bbox_inches="tight")
    plt.show()


def make_supplfig1():
    df = load_and_prep_data()

    fig, axes = plt.subplots(1, 3, figsize=(10, 6))
    axA, axB, axC = axes

    # =========================
    # Panel A: symptom trajectory over all weeks
    # =========================
    session_order = ["pre", "week 1", "week 2", "week 3", "week 4", "post"]

    hads_long = (
        df.groupby(["patient", "session"])["hads_dep_total"]
        .first()
        .reset_index()
        .dropna(subset=["hads_dep_total"])
    )
    hads_long = hads_long[hads_long["session"].isin(session_order)].copy()
    hads_long["session"] = pd.Categorical(
        hads_long["session"], categories=session_order, ordered=True
    )

    model = smf.mixedlm(
        f"hads_dep_total ~ session",
        data=hads_long,
        groups=hads_long["patient"]
    ).fit()

    print(model.summary())

    sns.lineplot(
        data=hads_long,
        x="session",
        y="hads_dep_total",
        estimator="mean",
        errorbar="ci",
        marker="o",
        linewidth=3,
        color=base_color,
        ax=axA,
    )
    # set tick labels properly
    axA.set_xticklabels(["Pre", "1", "2", "3", "4", "Post"], fontsize=11)
    axA.set_xlabel("Week")
    axA.set_ylabel("HADS-D")
    axA.set_title("a", fontsize=20)
    axA.spines["top"].set_visible(False)
    axA.spines["right"].set_visible(False)

    # =========================
    # Panel B: baseline predictors of week 4 HADS-D
    # =========================
    wide = (
        df.pivot_table(index="patient", columns="session", values="hads_dep_total", aggfunc="first")
    )

    baseline_vars = (
        df[df["session"] == "pre"]
        .drop_duplicates("patient")
        .set_index("patient")[
            ["age", "gender", "years_with_depression", "previous ect", "previous tms",
             "research tier"]
        ]
        .rename(columns={
            "previous ect": "previous_ect",
            "previous tms": "previous_tms",
            "research tier": "research_tier",
        })
    )

    model_df = wide.join(baseline_vars, how="inner")
    model_df = model_df.rename(columns={"pre": "baseline_hads", "week 4": "week4_hads"})
    model_df = model_df.dropna(subset=["week4_hads", "baseline_hads"])

    categorical_vars = ["gender", "previous_ect", "previous_tms", "research_tier"]
    for col in categorical_vars:
        if col in model_df.columns:
            model_df[col] = model_df[col].astype("category")

    continuous_vars = ["baseline_hads", "age", "age_onset", "treatment_days_num"]
    for col in continuous_vars:
        if col in model_df.columns:
            model_df[col] = (model_df[col] - model_df[col].mean()) / model_df[col].std()

    model = smf.ols(
        "week4_hads ~ baseline_hads + age + gender + previous_ect + previous_tms + research_tier",
        data=model_df,
    ).fit()

    coefs = model.params.drop("Intercept")
    conf = model.conf_int().loc[coefs.index]
    coefs_df = pd.DataFrame({
        "Predictor": coefs.index,
        "coef": coefs.values,
        "lower": conf[0].values,
        "upper": conf[1].values,
    })

    rename_dict = {
        "baseline_hads": "Baseline HADS-D",
        "age": "Age",
        "gender[T.Male]": "Male",
        "previous_ect[T.Yes]": "Previous ECT",
        "previous_tms[T.Yes]": "Previous TMS",
        "research_tier[T.Naturalistic]": "Naturalistic",
        "research_tier[T.RCT acceptable]": "RCT acceptable",
    }
    coefs_df["Predictor"] = coefs_df["Predictor"].replace(rename_dict)
    coefs_df = coefs_df.sort_values("coef")

    axB.axvline(0, color="gray", linestyle="--", linewidth=1)
    axB.scatter(coefs_df["coef"], range(len(coefs_df)), color="black", zorder=3)
    for i, row in enumerate(coefs_df.itertuples()):
        axB.plot([row.lower, row.upper], [i, i], color="black", linewidth=2)

    axB.set_yticks(range(len(coefs_df)))
    axB.set_yticklabels(coefs_df["Predictor"])
    axB.set_xlabel("Regression coefficient (β)")
    axB.set_ylabel("")
    axB.set_title("b", fontsize=20)
    axB.spines["top"].set_visible(False)
    axB.spines["right"].set_visible(False)

    # =========================
    # Panel C: HADS-D improvement vs MADRS improvement
    # =========================
    clinical_wide = df.pivot_table(
        index="patient",
        columns="session",
        values=["hads_dep_total", "madrs_total"],
        aggfunc="first",
    )

    hads_pre = clinical_wide[("hads_dep_total", "pre")]
    hads_post = clinical_wide[("hads_dep_total", "post")]
    madrs_pre = clinical_wide[("madrs_total", "pre")]
    madrs_post = clinical_wide[("madrs_total", "post")]

    corr_df = pd.DataFrame({
        "HADS-D improvement": hads_pre - hads_post,
        "MADRS improvement": madrs_pre - madrs_post,
    }).dropna()

    sns.regplot(
        data=corr_df,
        x="HADS-D improvement",
        y="MADRS improvement",
        scatter_kws={"s": 50, "alpha": 0.8, "color": base_color},
        line_kws={"color": "black", "linewidth": 2},
        ax=axC,
    )

    rho, pval = spearmanr(
                            corr_df["HADS-D improvement"],
                            corr_df["MADRS improvement"],)
    axC.axhline(0, color="gray", linestyle="--", linewidth=1)
    axC.axvline(0, color="gray", linestyle="--", linewidth=1)
    axC.text(
        0.05, 0.95,
        f"Spearman ρ = {rho:.2f}\np = {pval:.3g}\nN = {len(corr_df)}",
        transform=axC.transAxes,
        ha="left", va="top"
    )

    axC.set_title("c", fontsize=20)
    axC.spines["top"].set_visible(False)
    axC.spines["right"].set_visible(False)

    plt.tight_layout()
    plt.savefig(fig_dir_suppl / "supplfig1.png", dpi=300, bbox_inches="tight")
    plt.savefig(fig_dir_suppl / "supplfig1.svg", dpi=300, bbox_inches="tight")
    plt.show()

    print(model.summary())
    print(f"Spearman rho (HADS-D vs MADRS improvement): {rho:.3f}")