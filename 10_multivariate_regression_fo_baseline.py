"""
Baseline FO vs depressive symptoms (HADS-D) across HMM states.

Workflow:
1) Load clinical + HMM summary data (hmm_demo_quest_{n_states}.csv)
2) Subset baseline: Session 1, pre-TMS
3) Logit-transform FO (with clipping)
4) Fit per-state OLS: fo_logit ~ HADS-D + age + gender (robust SE)
5) Multiple-comparisons correction across fitted states (optional)
6) Plot: FO distribution, forest plot, regression prediction-style panels, predicted-vs-observed
"""
# Disclaimer: Code was prettified using AI

# The author still takes full responsibility for the validity of any logic and results

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import statsmodels.formula.api as smf
from statsmodels.stats.multitest import multipletests


# -----------------------------
# Plot style (Nature-ish)
# -----------------------------
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams.update({
    "font.family": "Arial",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})


# -----------------------------
# Utilities
# -----------------------------
def get_base_dir(system: str) -> Path:
    if system == "linux":
        return Path("/home/carinaf/LabData")
    if system == "windows":
        return Path("L:")
    raise ValueError("system must be 'windows' or 'linux'")


def logit_transform_fo(fo: pd.Series, eps: float = 1e-4) -> pd.Series:
    """Logit transform with clipping to avoid infinities."""
    x = fo.astype(float).clip(eps, 1 - eps)
    return np.log(x / (1 - x))


def inv_logit(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def load_and_prep_data(hmm_dir: Path, n_states: int, exclude_repeater: bool = False,
                        exclude_new_protocol: bool = False) -> pd.DataFrame:
    """
    Load and preprocess HMM demo questionnaire data.
    - shifts state labels to start at 1
    - fills demographics per patient
    - casts common columns to categorical
    """
    csv_path = Path(f'{hmm_dir}/hmm_demo_hads2704_{n_states}.csv')
    df = pd.read_csv(csv_path)

    if exclude_repeater and "patient" in df.columns:
        df = df[~df["patient"].astype(str).str.contains("R")]

    # control analyis: check if protocol changes results
    patient_num = df['patient'].astype(str).str.extract(r'(\d+)').astype(int)[0]

    base_ids = ['180', '183']

    df['protocol'] = np.where(
                        df['patient'].astype(str).isin(base_ids) | (patient_num >= 186),
            2,
            1
        )
    
    if exclude_new_protocol:
        df = df[
            ~(
                df['patient'].isin(base_ids) |
                (patient_num >= 186)
            )
        ]

    print(f"Analyzing {df['patient'].nunique()} patients (n_states={n_states})")

    df['eeg_date'] = pd.to_datetime(df['visit_anchor_datetime'])
    df['matched_hads_date'] = pd.to_datetime(df['matched_hads_date'])

    # add delta days between eeg and hads
    df['diff_days'] = (df['eeg_date'] - df['matched_hads_date']).dt.days

    df = add_recording_length(df)

    # states start at 1
    if "state" in df.columns:
        df["state"] = df["state"] + 1

    # fill demographics (first value per patient)
    for col in ["age", "gender", "responder", "group", "years_with_depression"]:
        if col in df.columns:
            df[col] = df.groupby("patient")[col].transform("first")

    # categorical conversion (safe)
    for col in ["patient", "session", "tms", "state", "responder", "group", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category", errors="ignore")

    return df


def get_baseline_df(df: pd.DataFrame) -> pd.DataFrame:
    """Baseline subset: visit 1, pre-TMS. Drops patients without HADS-D."""

    d = df.copy()

    # use visit-based logic (new pipeline)
    if "session" not in d.columns or "tms" not in d.columns:
        raise ValueError("Expected 'session' and 'tms' columns.")

    d_visit = d["session"].astype(str)
    d_tms = d["tms"].astype(str)

    baseline = d[(d_visit == "1") & (d_tms == "pre")].copy()

    # ---------------------------------------------------
    # 🚨 Drop patients with missing baseline HADS-D
    # ---------------------------------------------------
    missing_mask = baseline["hads_dep_total"].isna()

    if missing_mask.any():
        missing_patients = baseline.loc[missing_mask, "patient"].unique()

        print("\n🚨 Dropping patients with NO baseline HADS-D:")
        print(", ".join(map(str, missing_patients)))
        print(f"Total dropped: {len(missing_patients)}\n")

        #remove those patients entirely
        baseline = baseline[~baseline["patient"].isin(missing_patients)].copy()

    return baseline


def fit_statewise_baseline_models(
    df_baseline: pd.DataFrame,
    robust: str = "HC3",
    min_n: int = 10,
) -> Tuple[Dict[int, object], pd.DataFrame]:
    """
    Fit OLS per state:
      fo_logit ~ hads_dep_total + age + C(gender)
    Returns:
      models: dict[state] -> fitted model
      summary_df: state-wise table (beta, SE, CI, p, n)
    """
    required = ["state", "fo", "hads_dep_total", "age", "gender", 'eeg_recording_length']
    missing = [c for c in required if c not in df_baseline.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    d = df_baseline.copy()

    # Ensure state is numeric-ish for ordering
    d["state_num"] = pd.to_numeric(d["state"].astype(str), errors="coerce")

    # logit FO
    d["fo_logit"] = logit_transform_fo(d["fo"])

    models: Dict[int, object] = {}
    rows: list[dict] = []

    for st in sorted(d["state_num"].dropna().unique().astype(int).tolist()):
        ds = d[d["state_num"] == st].copy()
        if len(ds) < min_n:
            continue

        m = smf.ols("fo_logit ~ hads_dep_total + age + C(gender)", data=ds).fit(cov_type=robust)
        models[st] = m

        term = "hads_dep_total"
        ci = m.conf_int(alpha=0.05)

        rows.append({
            "state": int(st),
            "n": int(len(ds)),
            "beta": float(m.params[term]),
            "se": float(m.bse[term]),
            "ci_low": float(ci.loc[term, 0]),
            "ci_high": float(ci.loc[term, 1]),
            "p": float(m.pvalues[term]),
        })

    summary_df = pd.DataFrame(rows).sort_values("state").reset_index(drop=True)
    return models, summary_df


def fit_state_interaction_model(
    df_baseline: pd.DataFrame,
    states: Tuple[int, int] = (1, 2),
    cluster_by_patient: bool = True,
):
    """
    Test whether the HADS-D effect on FO differs between two states.

    Model:
        fo_logit ~ hads_dep_total * C(state_num) + age + C(gender)

    Parameters
    ----------
    df_baseline : pd.DataFrame
        Baseline dataframe (session 1, pre-TMS).
    states : tuple of int
        Two states to compare.
    cluster_by_patient : bool
        If True, use cluster-robust SE by patient because each patient contributes
        one observation per state.

    Returns
    -------
    model : statsmodels result
        Fitted interaction model.
    dsub : pd.DataFrame
        Data used for the model.
    """
    required = ["patient", "state", "fo", "hads_dep_total", "age", "gender", 'eeg_recording_length']
    missing = [c for c in required if c not in df_baseline.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    s1, s2 = states
    dsub = df_baseline.copy()
    dsub["state_num"] = pd.to_numeric(dsub["state"].astype(str), errors="coerce")
    dsub = dsub[dsub["state_num"].isin([s1, s2])].copy()
    dsub["state_num"] = dsub["state_num"].astype(int)
    dsub["fo_logit"] = logit_transform_fo(dsub["fo"])

    # set first state as reference
    dsub["state_num"] = pd.Categorical(dsub["state_num"], categories=[s1, s2], ordered=True)

    formula = "fo_logit ~ hads_dep_total * C(state_num) + age + C(gender)"

    if cluster_by_patient:
        model = smf.ols(formula, data=dsub).fit(
            cov_type="cluster",
            cov_kwds={"groups": dsub["patient"]}
        )
    else:
        model = smf.ols(formula, data=dsub).fit(cov_type="HC3")

    print(model.summary())
    m_mixed = smf.mixedlm(
        "fo_logit ~ hads_dep_total * C(state_num) + age + C(gender)",
        data=dsub,
        groups=dsub["patient"]
    ).fit(method='powell')

    print(m_mixed.summary())

    return model, dsub


def apply_pvalue_correction(summary_df: pd.DataFrame, correction: Optional[str], alpha: float = 0.05) -> pd.DataFrame:
    """
    Add p_corr and sig columns.
    correction: e.g., 'bonferroni', 'fdr_bh', or None to skip.
    """
    out = summary_df.copy()
    if len(out) == 0:
        out["p_corr"] = []
        out["sig"] = []
        return out

    if correction:
        out["p_corr"] = multipletests(out["p"].values, method=correction)[1]
    else:
        out["p_corr"] = out["p"].values

    out["sig"] = out["p_corr"] < alpha
    return out


# -----------------------------
# Plotting
# -----------------------------
def _strip_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")


def plot_baseline_figure(
    df_baseline: pd.DataFrame,
    models: Dict[int, object],
    summary_df: pd.DataFrame,
    n_states: int,
    state_for_panels: Tuple[int, int] = (1, 2),
    figsize=(12, 12),
    out_base: Optional[Path] = None,
    plot_preds: bool = False
):
    """
    Creates 3x2 figure:
      A violin+box of FO across states
      B forest plot of beta(HADS-D) per state
      C-D prediction-style regression for selected states
      E-F predicted vs observed FO for selected states
    """
    # Seaborn palette for consistent state coloring
    colors = sns.color_palette("tab20", n_colors=max(n_states, 2))
    state_color_map = {st: colors[st - 1] if (st - 1) < len(colors) else "black" for st in range(1, n_states + 1)}
    state_order = list(range(1, n_states + 1))

    d = df_baseline.copy()
    d["state_num"] = pd.to_numeric(d["state"].astype(str), errors="coerce").astype("Int64")
    d = d[d["state_num"].notna()].copy()
    d["state_num"] = d["state_num"].astype(int)
    d["state_cat"] = pd.Categorical(d["state_num"], categories=state_order, ordered=True)

    fig, axes = plt.subplots(
        3, 2,
        figsize=figsize,
        gridspec_kw={"height_ratios": [1.2, 1, 1]},
        constrained_layout=True
    )
    axA, axB = axes[0, 0], axes[0, 1]
    axC, axD = axes[1, 0], axes[1, 1]
    axE, axF = axes[2, 0], axes[2, 1]

    # ---- Panel A: FO distribution across states ----
    sns.violinplot(
        data=d,
        x="state_cat",
        y="fo",
        order=state_order,
        palette=colors,
        cut=0,
        inner=None,
        linewidth=1,
        ax=axA
    )
    sns.boxplot(
        data=d,
        x="state_cat",
        y="fo",
        order=state_order,
        width=0.25,
        showcaps=False,
        showfliers=False,
        boxprops={"facecolor": "none", "edgecolor": "black", "linewidth": 1.4},
        whiskerprops={"color": "black", "linewidth": 1.2},
        capprops={"color": "black", "linewidth": 1.2},
        medianprops={"color": "black", "linewidth": 2.2},
        ax=axA
    )
    axA.set_xlabel("HMM states")
    axA.set_ylabel("Fractional occupancy (FO)")
    _strip_spines(axA)

    # ---- Panel B: forest plot of beta(HADS-D) ----
    if len(summary_df) > 0:
        y = np.arange(len(summary_df))[::-1]
        labels = summary_df["state"].astype(str).tolist()

        for yi, (_, row) in zip(y, summary_df.iterrows()):
            st = int(row["state"])
            c = state_color_map.get(st, "black")
            axB.plot([row["ci_low"], row["ci_high"]], [yi, yi], color=c, linewidth=2)
            axB.scatter(row["beta"], yi, s=55, facecolor=c, edgecolor=c, linewidth=1.0, zorder=3)

        axB.axvline(0, color="black", linewidth=1)
        axB.set_yticks(y)
        axB.set_yticklabels(labels)
        axB.set_ylabel("HMM states")
        axB.set_xlabel("β (HADS-D) predicting FO (logit scale)")
    else:
        axB.text(0.5, 0.5, "No states met min_n", ha="center", va="center", transform=axB.transAxes)
        axB.set_axis_off()
    _strip_spines(axB)

    # ---- Panel C/D: prediction-style regression for selected states ----
    def plot_state_reg(ax, st: int):
        ds = d[d["state_num"] == st].copy()
        if st not in models or len(ds) < 10:
            ax.text(0.5, 0.5, f"State {st}\n(n<10)", ha="center", va="center", transform=ax.transAxes)
            ax.set_axis_off()
            return

        m = models[st]

        # scatter FO vs HADS-D
        ax.scatter(ds["hads_dep_total"], ds["fo"], alpha=0.5, s=18, color="black")

        x_vals = np.linspace(float(ds["hads_dep_total"].min()), float(ds["hads_dep_total"].max()), 200)
        pred_df = pd.DataFrame({
            "hads_dep_total": x_vals,
            "age": float(ds["age"].mean()),
            "gender": ds["gender"].mode().iloc[0],
        })

        pred = m.get_prediction(pred_df)
        mean_logit = pred.predicted_mean
        ci = pred.conf_int(alpha=0.05)

        mean = inv_logit(mean_logit)
        low = inv_logit(ci[:, 0])
        high = inv_logit(ci[:, 1])

        ax.plot(x_vals, mean, color="black", linewidth=3)
        ax.fill_between(x_vals, low, high, color="black", alpha=0.2)
        ax.set_xlabel("Baseline HADS-D")
        ax.set_ylabel("FO")
        _strip_spines(ax)

    plot_state_reg(axC, int(state_for_panels[0]))
    plot_state_reg(axD, int(state_for_panels[1]))

    if plot_preds:
        # ---- Panel E/F: predicted vs observed FO ----
        def plot_pred_vs_obs(ax, st: int):
            ds = d[d["state_num"] == st].copy()
            if st not in models or len(ds) < 10:
                ax.text(0.5, 0.5, f"State {st}\n(n<10)", ha="center", va="center", transform=ax.transAxes)
                ax.set_axis_off()
                return

            m = models[st]
            pred = m.get_prediction(ds).predicted_mean
            pred_fo = inv_logit(pred)
            obs_fo = ds["fo"].astype(float).values

            ax.scatter(pred_fo, obs_fo, alpha=0.6, s=18, color="black")

            lim_low = 0.0
            lim_high = float(np.nanmax([pred_fo.max(), obs_fo.max()]))
            ax.plot([lim_low, lim_high], [lim_low, lim_high], linestyle="--", color="black", linewidth=1)

            ax.set_xlabel("Predicted FO")
            ax.set_ylabel("Observed FO")
            _strip_spines(ax)

        plot_pred_vs_obs(axE, int(state_for_panels[0]))
        plot_pred_vs_obs(axF, int(state_for_panels[1]))

    if out_base is not None:
        fig.savefig(str(out_base) + ".svg")
        fig.savefig(str(out_base) + ".png", dpi=300)

    return fig, axes


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


def run_baseline_fo_pipeline(
    hmm_dir: Path,
    fig_dir: Path,
    n_states: int,
    state_for_panels: Tuple[int, int] = (1, 2),
    correction: Optional[str] = "bonferroni",
    alpha: float = 0.05,
    robust: str = "HC3",
    min_n: int = 10,
):
    df = load_and_prep_data(hmm_dir, n_states=n_states)
    dfb = get_baseline_df(df) # 700 rows, 70 patients, 10 states
    models, summary_df = fit_statewise_baseline_models(dfb, robust=robust, min_n=min_n)
    summary_df = apply_pvalue_correction(summary_df, correction=correction, alpha=alpha)

    out_base = fig_dir / f"baseline_fo_regression_states{n_states}"
    plot_baseline_figure(
        df_baseline=dfb,
        models=models,
        summary_df=summary_df,
        n_states=n_states,
        state_for_panels=state_for_panels,
        out_base=out_base,
    )

    return summary_df, models


if __name__ == "__main__":
    system = "windows"
    base_dir = get_base_dir(system)

    hmm_dir = base_dir / "Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered"
    fig_dir = hmm_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    eeg_length_path = Path(
    f"{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/"
    "prepared_data_giles_05Hz_1Hzfiltereddata/eeg_recording_length.csv")

    # test robustness across different state solutions
    state_solutions = [10]

    for ns in state_solutions:
        summary, models = run_baseline_fo_pipeline(
            hmm_dir=hmm_dir,
            fig_dir=fig_dir,
            n_states=ns,
            state_for_panels=(1, 2),
            correction="bonferroni",   # set to None to skip correction
            robust="HC3",
            min_n=10,
        )
        print(summary)