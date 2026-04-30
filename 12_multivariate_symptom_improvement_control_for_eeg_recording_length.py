"""Run regressions on HMM states with HADS-D as dependent variable (depressive symptom severity)
# Does FO change predict changes in depressive symptoms?

Author: Carina Forster
"""
import pandas as pd
import numpy as np
from pathlib import Path

import statsmodels.formula.api as smf
import scipy.stats
import matplotlib.pyplot as plt

from statsmodels.stats.multitest import multipletests

# setting for nature publishing
plt.rcParams['pdf.fonttype'] = 42
# text editable in inkscape
plt.rcParams['svg.fonttype'] = 'none'

# linux doesn't have Arial
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

# run in base python (3.12)
system = 'windows'

if system == 'linux':
    base_dir = Path('/home/carinaf/LabData')
elif system == 'windows':
    base_dir = Path("L:")
else:
    raise ValueError("No available system path defined. Use 'windows' or 'linux'.")

# ------------ Directories -------------#
hmm_dir = Path(f'{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered')
fig_dir = Path(f"{hmm_dir}/figures")

# recording length file
eeg_length_path = Path(
    f"{base_dir}/Lab_LucaC/Carina/canonical_hmm_finalsample/"
    "prepared_data_giles_05Hz_1Hzfiltereddata/eeg_recording_length.csv"
)


def load_and_prep_data(n_states, exclude_repeater: bool = False):
    """
    Load and preprocess HMM demo questionnaire data.
    Drops patients with missing baseline HADS-D (session 1, pre).
    Adds EEG recording length by patient and session.
    """
    csv_path = Path(f"{hmm_dir}/hmm_demo_hads2704_{n_states}.csv")
    df = pd.read_csv(csv_path)

    df = add_recording_length(df)

    if exclude_repeater and "patient" in df.columns:
        df = df[~df["patient"].astype(str).str.contains("R")]

    if "state" in df.columns:
        df["state"] = df["state"] + 1

    for col in ["age", "gender", "responder", "group", "years_with_depression"]:
        if col in df.columns:
            df[col] = df.groupby("patient")[col].transform("first")

    # Drop patients with no baseline HADS-D
    required = {"session", "tms", "hads_dep_total", "patient"}
    if required.issubset(df.columns):
        baseline = df[
            (df["session"].astype(str) == "1") &
            (df["tms"].astype(str) == "pre")
        ].copy()

        missing_baseline = baseline.loc[
            baseline["hads_dep_total"].isna(), "patient"
        ].astype(str).unique()

        if len(missing_baseline) > 0:
            print("\nDropping patients with no baseline HADS-D before treatment:")
            print(", ".join(sorted(missing_baseline)))
            df = df[~df["patient"].astype(str).isin(missing_baseline)].copy()

    print(f"Analyzing {df['patient'].nunique()} patients")

    for col in ["patient", "session", "tms", "state", "responder", "group", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category", errors="ignore")

    return df


def prepare_delta_fo(
    n_states: int,
    state_for_reg=(1, 2),
    sessions=(1, 2),
    symptom_col: str = "hads_dep_total",
    reml: bool = True,
    scale_deltafo_by_100: bool = True,         # interpret ΔFO in percentage points
    scale_eeg_recording_length: bool = True,   # z-score recording length
):
    df = load_and_prep_data(n_states)

    st1, st2 = state_for_reg
    dfo1 = f"delta_fo_state{st1}"
    dfo2 = f"delta_fo_state{st2}"

    # Symptoms wide
    sym_wide = (
        df.pivot_table(index="patient", columns="session", values=symptom_col)
        .reset_index()
        .rename(columns={s: f"sym_s{s}" for s in df["session"].cat.categories})
    )

    cov = df[["patient", "age", "gender"]].drop_duplicates()

    # Delta FO (pre - post)
    fo_prepost = (
        df[df["tms"].isin(["pre", "post"])]
        .groupby(["patient", "session", "state", "tms"])["fo"]
        .mean()
        .reset_index()
    )

    fo_wide_tms = (
        fo_prepost
        .pivot_table(index=["patient", "session", "state"], columns="tms", values="fo")
        .reset_index()
    )

    if "pre" not in fo_wide_tms.columns or "post" not in fo_wide_tms.columns:
        raise ValueError("Expected both 'pre' and 'post' in tms to compute delta_fo = pre - post.")

    fo_wide_tms["delta_fo"] = fo_wide_tms["pre"].astype(float) - fo_wide_tms["post"].astype(float)
    fo_wide_tms = fo_wide_tms[fo_wide_tms["state"].isin(state_for_reg)].copy()

    # Stack transitions into long_df
    long_rows = []
    for s in sessions:
        base_col = f"sym_s{s}"
        next_col = f"sym_s{s+1}"
        if base_col not in sym_wide.columns or next_col not in sym_wide.columns:
            raise ValueError(f"Missing symptom sessions for transition {s}->{s+1} in symptom_col={symptom_col}")

        fo_sess = fo_wide_tms[fo_wide_tms["session"] == s].copy()
        fo_sess_wide = (
            fo_sess.pivot(index="patient", columns="state", values="delta_fo")
            .reset_index()
            .rename(columns={st1: dfo1, st2: dfo2})
        )

        # use recording length from the baseline session of the transition (pre TMS)
        #rec_len = (
        #    df[(df["session"].astype(int) == s) & (df["tms"] == 'pre')][["patient", "eeg_recording_length"]]
         #   .drop_duplicates()
        #)

        # use mean recording length of pre + post EEG
        rec_len = (
            df[df["session"].astype(int) == s]
            .groupby(["patient", "session"], as_index=False)["eeg_recording_length"]
            .mean()
        )

        tmp = (
            fo_sess_wide
            .merge(sym_wide[["patient", base_col, next_col]], on="patient", how="inner")
            .merge(rec_len, on="patient", how="left")
            .merge(cov, on="patient", how="left")
            .rename(columns={base_col: "sym_base", next_col: "sym_next"})
        )
        tmp["transition"] = s  # 1 means 1->2, 2 means 2->3
        long_rows.append(tmp)

    long_df = pd.concat(long_rows, ignore_index=True).dropna().copy()
    long_df["transition"] = long_df["transition"].astype("category")

    # Optional scaling to percentage points
    if scale_deltafo_by_100:
        long_df[dfo1] = long_df[dfo1].astype(float) * 100.0
        long_df[dfo2] = long_df[dfo2].astype(float) * 100.0

    # Optional z-scoring of recording length
    if scale_eeg_recording_length:
        mean_len = long_df["eeg_recording_length"].mean()
        std_len = long_df["eeg_recording_length"].std()
        if std_len > 0:
            long_df["eeg_recording_length"] = (
                (long_df["eeg_recording_length"] - mean_len) / std_len
            )
        else:
            print("Warning: eeg_recording_length has zero variance; not scaling.")

    return long_df


def fit_two_session_models(long_df, state_for_reg=(1, 2), robust="HC3"):

    st1, st2 = state_for_reg
    dfo1 = f"delta_fo_state{st1}"
    dfo2 = f"delta_fo_state{st2}"

    models = {}

    for sess in [1, 2]:
        d = long_df[long_df["transition"].astype(int) == sess].copy()

        model = smf.ols(
            f"sym_next ~ {dfo1} + {dfo2} + sym_base + age + eeg_recording_length + C(gender)",
            data=d
        ).fit(cov_type=robust)

        models[sess] = model

    return models


def plot_two_session_regression(
    long_df,
    models,
    state_for_reg=(1, 2),
    figsize=(12, 10),
    n_grid=200,
    savepath=None,
):
    st1, st2 = state_for_reg
    dfo1 = f"delta_fo_state{st1}"
    dfo2 = f"delta_fo_state{st2}"

    dfp = long_df.copy()
    dfp["improve_obs"] = dfp["sym_base"].astype(float) - dfp["sym_next"].astype(float)

    xlims = {
        dfo1: (float(dfp[dfo1].min()), float(dfp[dfo1].max())),
        dfo2: (float(dfp[dfo2].min()), float(dfp[dfo2].max())),
    }

    fig, axes = plt.subplots(
        2, 2,
        figsize=figsize,
        constrained_layout=True,
        sharex="col"
    )

    panels = [
        (0, dfo1, dfo2, f"State {st1}"),
        (1, dfo2, dfo1, f"State {st2}"),
    ]

    def plot_panel(ax, dsub, model, x_term, other_term, title=None):
        ax.scatter(dsub[x_term], dsub["improve_obs"], alpha=0.5, s=18, color="black")

        x_min, x_max = xlims[x_term]
        x_vals = np.linspace(x_min, x_max, n_grid)

        pred_df = pd.DataFrame({
            x_term: x_vals,
            other_term: float(dsub[other_term].mean()),
            "sym_base": float(dsub["sym_base"].mean()),
            "age": float(dsub["age"].mean()),
            "eeg_recording_length": float(dsub["eeg_recording_length"].mean()),
            "gender": dsub["gender"].mode().iloc[0],
        })

        pred = model.get_prediction(pred_df)
        yhat_next = pred.predicted_mean
        ci_next = pred.conf_int(alpha=0.05)

        sym_base_fixed = float(dsub["sym_base"].mean())
        mean_improve = sym_base_fixed - yhat_next
        low_improve = sym_base_fixed - ci_next[:, 1]
        high_improve = sym_base_fixed - ci_next[:, 0]

        ax.plot(x_vals, mean_improve, color="black", linewidth=3)
        ax.fill_between(x_vals, low_improve, high_improve, color="black", alpha=0.2)

        ax.axhline(0, linestyle=":", color="black", linewidth=1)
        ax.set_xlim(x_min, x_max)
        if title is not None:
            ax.set_title(title)

        ax.set_xlabel("% ΔFO (pre − post TMS)")
        ax.set_ylabel("Predicted symptom improvement")

    for row, sess in enumerate([1, 2]):
        dsub = dfp[dfp["transition"].astype(int) == sess].copy()
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

    for ax in axes.flatten():
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_linewidth(1.5)
        ax.spines["bottom"].set_linewidth(1.5)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    if savepath:
        fig.savefig(savepath + ".png", dpi=300)
        fig.savefig(savepath + ".svg")

    return fig, axes


def plot_pre_post_boxplots_with_sig(
    n_states: int,
    states=(1, 2),
    correction="bonferroni",
    alpha=0.05,
    figsize=(10, 6),
    fig_dir=".",
    show_pairs=True,
):
    df = load_and_prep_data(n_states)
    df = df[df["state"].isin(states)].copy()

    paired_rows = []
    test_rows = []

    sessions = sorted(df["session"].dropna().unique())

    for sess in sessions:
        for st in states:
            d = df[(df["session"] == sess) & (df["state"] == st)].copy()

            wide = (
                d.pivot_table(index="patient", columns="tms", values="fo")
                .dropna()
            )

            if not {"pre", "post"}.issubset(wide.columns):
                continue

            pre = wide["pre"].astype(float)
            post = wide["post"].astype(float)

            if len(pre) < 5:
                continue

            tval, pval = scipy.stats.ttest_rel(post, pre)

            test_rows.append({
                "session": int(sess),
                "state": int(st),
                "n": int(len(pre)),
                "t": float(tval),
                "p": float(pval),
                "mean_diff": float((post - pre).mean())
            })

            for pid in wide.index:
                paired_rows.append({
                    "session": int(sess),
                    "state": int(st),
                    "patient": pid,
                    "pre": float(wide.loc[pid, "pre"]),
                    "post": float(wide.loc[pid, "post"]),
                })

    tests_df = pd.DataFrame(test_rows)
    paired_df = pd.DataFrame(paired_rows)

    if correction and len(tests_df) > 0:
        tests_df["p_corr"] = multipletests(tests_df["p"], method=correction)[1]
    else:
        tests_df["p_corr"] = tests_df["p"]

    tests_df["sig"] = tests_df["p_corr"] < alpha

    fig, axes = plt.subplots(
        len(sessions), len(states),
        figsize=figsize,
        constrained_layout=True
    )
    if len(sessions) == 1 and len(states) == 1:
        axes = np.array([[axes]])
    elif len(sessions) == 1:
        axes = np.array([axes])
    elif len(states) == 1:
        axes = np.array([[ax] for ax in axes])

    def p_to_stars(p):
        if not np.isfinite(p):
            return "n/a"
        if p < 0.001:
            return "***"
        if p < 0.01:
            return "**"
        if p < 0.05:
            return "*"
        return "n.s."

    def add_sig_bracket(ax, x1, x2, y, text):
        h = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.03
        ax.plot([x1, x1, x2, x2], [y, y + h, y + h, y], linewidth=1.2, color="black")
        ax.text((x1 + x2) / 2, y + h * 1.2, text, ha="center", va="bottom")

    for i, sess in enumerate(sessions):
        for j, st in enumerate(states):
            ax = axes[i, j]
            d = paired_df[(paired_df["session"] == sess) & (paired_df["state"] == st)].copy()

            if len(d) == 0:
                ax.set_axis_off()
                continue

            pre_vals = d["pre"].values
            post_vals = d["post"].values

            ax.boxplot(
                [pre_vals, post_vals],
                positions=[0, 1],
                widths=0.55,
                patch_artist=True,
                boxprops=dict(facecolor="white", edgecolor="black"),
                medianprops=dict(color="black", linewidth=2),
                whiskerprops=dict(color="black"),
                capprops=dict(color="black"),
                flierprops=dict(marker="o", markersize=3, markerfacecolor="black",
                                markeredgecolor="black", alpha=0.4),
            )

            if show_pairs:
                for _, r in d.iterrows():
                    ax.plot([0, 1], [r["pre"], r["post"]], color="gray", alpha=0.35, linewidth=1)
                ax.scatter(np.zeros(len(d)), pre_vals, s=14, alpha=0.6, color="black")
                ax.scatter(np.ones(len(d)), post_vals, s=14, alpha=0.6, color="black")

            ax.set_xticks([0, 1])
            ax.set_xticklabels(["Pre", "Post"])
            ax.set_title(f"Session {sess} – State {st} (n={len(d)})")
            ax.set_ylabel("FO")

            row = tests_df[(tests_df["session"] == sess) & (tests_df["state"] == st)]
            if len(row) > 0:
                p_corr = float(row["p_corr"].iloc[0])
                stars = p_to_stars(p_corr)

                y = float(np.nanmax([pre_vals.max(), post_vals.max()]))
                add_sig_bracket(ax, 0, 1, y * 1.05 if y > 0 else y + 0.05, stars)
                ax.text(0.5, 0.02, f"p_corr={p_corr:.3g}", ha="center", va="bottom", transform=ax.transAxes)

    fig.savefig(f"{fig_dir}/fo_pre_post_boxplots_states{states[0]}_{states[1]}.svg")
    fig.savefig(f"{fig_dir}/fo_pre_post_boxplots_states{states[0]}_{states[1]}.png", dpi=300)

    return fig, axes, tests_df, paired_df


def compare_interaction_model(long_df):
    from scipy.stats import chi2

    model_no_int = smf.mixedlm(
        "sym_next ~ delta_fo_state1 + delta_fo_state2 + transition + sym_base + age + eeg_recording_length + C(gender)",
        data=long_df,
        groups=long_df["patient"]
    ).fit(reml=False)

    model_int = smf.mixedlm(
        "sym_next ~ delta_fo_state1 * transition + delta_fo_state2 + sym_base + age + eeg_recording_length + C(gender)",
        data=long_df,
        groups=long_df["patient"]
    ).fit(reml=False)

    lr_stat = 2 * (model_int.llf - model_no_int.llf)
    df_diff = model_int.df_modelwc - model_no_int.df_modelwc
    p_value = chi2.sf(lr_stat, df_diff)

    print("Model comparison (Likelihood Ratio Test)")
    print("----------------------------------------")
    print(f"LR statistic: {lr_stat:.3f}")
    print(f"df difference: {df_diff}")
    print(f"p-value: {p_value:.4f}")

    return model_no_int, model_int, lr_stat, df_diff, p_value


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


def plot_transition2_control_betas(all_states, fig_dir, state_for_reg=(1, 2)):
    """
    Supplementary figure:
    A) beta for ΔFO state 1
    B) beta for ΔFO state 2
    C) beta for EEG recording length

    Uses transition 2 models only.
    """

    rows = []

    st1, st2 = state_for_reg
    terms = [
        f"delta_fo_state{st1}",
        f"delta_fo_state{st2}",
        "eeg_recording_length",
    ]

    for n_states in all_states:
        print(f"Extracting betas for n_states={n_states}")

        long_df = prepare_delta_fo(
            n_states,
            state_for_reg=state_for_reg,
            scale_deltafo_by_100=True,
            scale_eeg_recording_length=True,
        )

        models = fit_two_session_models(
            long_df,
            state_for_reg=state_for_reg,
        )

        model = models[2]  # second transition only

        ci = model.conf_int(alpha=0.05)

        for term in terms:
            rows.append({
                "n_states": n_states,
                "term": term,
                "beta": model.params[term],
                "se": model.bse[term],
                "ci_low": ci.loc[term, 0],
                "ci_high": ci.loc[term, 1],
                "p": model.pvalues[term],
            })

    beta_df = pd.DataFrame(rows)
    beta_df.to_csv(fig_dir / "supp_transition2_control_betas.csv", index=False)

    # -----------------------------
    # Plot
    # -----------------------------
    panel_info = [
        (f"delta_fo_state{st2}", f"a  symptom change Session 11 - Session 20 HADS-D ~ ΔFO state {st2}"),
        ("eeg_recording_length", "b  symptom change Session 11 - Session 20 HADS-D ~ EEG recording length"),
    ]

    fig, axes = plt.subplots(
        1, 2,
        figsize=(12, 4),
        constrained_layout=True,
        sharex=True,
    )

    for ax, (term, title) in zip(axes, panel_info):
        d = beta_df[beta_df["term"] == term].copy()
        d = d.sort_values("n_states")

        x = np.arange(len(d))

        ax.errorbar(
            x,
            d["beta"],
            yerr=[
                d["beta"] - d["ci_low"],
                d["ci_high"] - d["beta"],
            ],
            fmt="o",
            color="black",
            ecolor="black",
            elinewidth=1.5,
            capsize=4,
        )

        ax.axhline(0, linestyle=":", color="black", linewidth=1)

        ax.set_title(title)
        ax.set_xlabel("HMM states")
        ax.set_xticks(x)
        ax.set_xticklabels(d["n_states"].astype(str))
        ax.set_ylabel("β coefficient")

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.savefig(fig_dir / "supp_transition2_control_betas.svg")
    fig.savefig(fig_dir / "supp_transition2_control_betas.png", dpi=300)

    return fig, axes, beta_df


if __name__ == "__main__":

    all_states = [6, 8, 10, 12]
    n_sessions = 6

    for n_states in all_states:
        print(f"{n_states}")
        long_df = prepare_delta_fo(
            n_states,
            scale_deltafo_by_100=True,
            scale_eeg_recording_length=True
        )

        print("\nMissing eeg_recording_length values:")
        print(long_df["eeg_recording_length"].isna().sum())

        models = fit_two_session_models(long_df)
        print(models[1].summary())
        print(models[2].summary())

        plot_two_session_regression(
            long_df,
            models,
            savepath=f'{fig_dir}/fig3_{n_states}'
        )
    
    # suppl. fig. 9: eeg_recording_length as control
    fig, axes, beta_df = plot_transition2_control_betas(
    all_states=all_states,
    fig_dir=fig_dir,
    state_for_reg=(1, 2),
    )