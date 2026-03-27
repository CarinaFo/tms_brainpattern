"""
PCA on HMM transition probability matrices and association with symptoms.

Workflow:
1) Load per-session transition probability matrices (tp_{ses}_{n_states}.npy) (length is n_participants)
2) Vectorize off-diagonal transitions and fit PCA (after z-scoring features)
3) Attach PCA scores (PC1-2) to a long dataframe indexed by (patient, session_0to5)
4) Merge with clinical data (hmm_demo_quest_{n_states}.csv) by (patient, session, tms)
5) Baseline: regress PC1/PC2 ~ baseline HADS-D + age + gender (Session 1 pre)
6) Change: compute ΔPC2 (pre - post) per session and regress next-session symptoms
"""

from pathlib import Path
import joblib

import numpy as np
import pandas as pd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

import statsmodels.formula.api as smf
from scipy.stats import zscore

import matplotlib.pyplot as plt


# -----------------------------
# Plot style (Nature-ish)
# -----------------------------
plt.rcParams["pdf.fonttype"] = 42
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

# -----------------------------
# Paths
# -----------------------------
system = "windows"  # "linux" or "windows"

if system == "linux":
    base_dir = Path("/home/carinaf/LabData")
elif system == "windows":
    base_dir = Path("L:")
else:
    raise ValueError("system must be 'windows' or 'linux'")

hmm_dir = base_dir / "Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered"
fig_dir = hmm_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

# -----------------------------
# Core PCA functions
# -----------------------------
def load_transitions(hmm_dir: Path, n_sessions: int, n_states: int) -> np.ndarray:
    """
    Load transition probability matrices into array:
    shape = (n_sessions, n_patients, n_states, n_states)
    """
    mats = []
    for ses in range(n_sessions):
        fp = hmm_dir / f"tp_{ses}_{n_states}.npy"
        mats.append(np.load(fp))
    transitions = np.stack(mats, axis=0)
    return transitions


def vectorize_offdiagonal(transitions: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Vectorize off-diagonal entries of each transition matrix.
    Returns:
      X_off: (n_samples, n_features_offdiag)
      mask:  (n_states, n_states) boolean mask used
    """
    n_sessions, n_patients, n_states, _ = transitions.shape
    mask = ~np.eye(n_states, dtype=bool)

    # reshape to (n_samples, n_states, n_states) then take off-diagonal features
    X_off = transitions.reshape(-1, n_states, n_states)[:, mask]
    return X_off, mask


def run_pca_on_transitions(
    hmm_dir: Path,
    n_sessions: int,
    n_states: int,
    standardize: bool = True,
) -> dict:
    """
    Fit PCA on off-diagonal transition probabilities.
    """
    transitions = load_transitions(hmm_dir, n_sessions=n_sessions, n_states=n_states)
    X_off, mask = vectorize_offdiagonal(transitions)

    scaler = None
    X_in = X_off
    if standardize:
        scaler = StandardScaler()
        X_in = scaler.fit_transform(X_off)

    pca = PCA(n_components=None)

    scores = pca.fit_transform(X_in)      # (n_samples, n_components)

    # save the PCA output to disk
    save_pca_results(
        pca,
        filepath=f'{hmm_dir}/pca_results_{n_states}states.joblib'
    )

    loadings = pca.components_            # (n_components, n_features_offdiag)

    cumvar = np.cumsum(pca.explained_variance_ratio_) * 100
    print("Explained variance ratio:", pca.explained_variance_ratio_)
    print("Cumulative variance (%):", cumvar)

    return {
        "transitions": transitions,
        "X_off": X_off,
        "mask": mask,
        "scaler": scaler,
        "pca": pca,
        "scores": scores,
        "loadings": loadings,
        "cumvar": cumvar,
    }


# -----------------------------
# Data merge helpers
# -----------------------------
def build_transition_scores_df(scores: np.ndarray, df_patients: pd.DataFrame, n_sessions: int) -> pd.DataFrame:
    """
    Build long dataframe with columns:
      patient, session_0to5, df_session(1..), tms(pre/post), PC1, PC2, PC3

    NOTE: This assumes that PCA rows correspond to:
      session_0to5 = 0..n_sessions-1, and within each session, patients are in `patients` order.
    """
    patients = pd.unique(df_patients["patient"])
    n_patients = len(patients)

    if scores.shape[0] != n_sessions * n_patients:
        raise ValueError(
            f"Score rows ({scores.shape[0]}) != n_sessions*n_patients ({n_sessions*n_patients}). "
            "This indicates a mismatch in ordering or patient count."
        )

    patient_ids = np.tile(patients, n_sessions)
    session_0to5 = np.repeat(np.arange(n_sessions), n_patients)

    df_scores = pd.DataFrame(scores[:, :3], columns=["PC1", "PC2", "PC3"])
    df_scores["patient"] = patient_ids
    df_scores["session_0to5"] = session_0to5

    # map to treatment session (1..3) and pre/post
    df_scores["df_session"] = (df_scores["session_0to5"] // 2) + 1
    df_scores["tms"] = np.where(df_scores["session_0to5"] % 2 == 0, "pre", "post")

    # types
    df_scores["patient"] = df_scores["patient"].astype("category")
    df_scores["df_session"] = df_scores["df_session"].astype("category")
    df_scores["tms"] = df_scores["tms"].astype("category")
    df_scores["session_0to5"] = df_scores["session_0to5"].astype("category")

    return df_scores


def load_clinical_df(hmm_dir: Path, n_states: int) -> pd.DataFrame:
    csv_path = hmm_dir / f"hmm_demo_quest_{n_states}.csv"
    df = pd.read_csv(csv_path)
    print(f"Analyzing {df['patient'].nunique()} patients")

    # fill demographics per patient (optional but usually useful)
    for col in ["age", "gender", "responder", "group", "years_with_depression"]:
        if col in df.columns:
            df[col] = df.groupby("patient")[col].transform("first")

    # categorical types
    for col in ["patient", "session", "tms", "state", "responder", "group", "gender"]:
        if col in df.columns:
            df[col] = df[col].astype("category", errors="ignore")

    return df


def add_pca_to_clinical(
    hmm_dir: Path,
    n_states: int,
    n_sessions: int = 6,
) -> tuple[pd.DataFrame, dict]:
    """
    Returns merged dataframe and PCA outputs dict.
    """
    pca_out = run_pca_on_transitions(hmm_dir, n_sessions=n_sessions, n_states=n_states, standardize=True)
    transitions = pca_out["transitions"]

    # plot mean transitions
    plot_transition_probs(transitions, outpath=fig_dir / "transitions_probs_mean.svg")

    # clinical dataframe
    df_clin = load_clinical_df(hmm_dir, n_states=n_states)

    # clinical vars are duplicated across state rows -> keep one row per patient/session/tms
    # choosing state==1 is a simple way if all clinical vars are state-invariant
    if "state" in df_clin.columns:
        df_clin_uniq = df_clin[df_clin["state"] == 2].copy()
    else:
        df_clin_uniq = df_clin.copy()

    # build scores df
    df_scores = build_transition_scores_df(pca_out["scores"], df_clin_uniq, n_sessions=n_sessions)

    df_merged = df_scores.merge(
        df_clin_uniq,
        left_on=["patient", "df_session", "tms"],
        right_on=["patient", "session", "tms"],
        how="inner",
        suffixes=("_pca", "_clin"),
    )

    # Keep ONE canonical session column: use clinical "session"
    # Rename the PCA-derived session to avoid duplication
    df_merged = df_merged.rename(columns={
        "df_session": "session_num",      # derived 1..3
        "session_0to5": "session_0to5",   # already fine
    })

    # Drop redundant session columns from the PCA side (if present)
    for c in ["session_pca"]:
        if c in df_merged.columns:
            df_merged = df_merged.drop(columns=[c])

    return df_merged, pca_out


# -----------------------------
# Analyses
# -----------------------------
def baseline_pc_vs_hads(df: pd.DataFrame, robust: str = "HC3"):
    """
    Baseline association: Session 1 pre only.
    """
    d = df.query("session == 1 and tms == 'pre'").copy()

    # outlier removal (PC2 + HADS-D)
    d = d[
        (np.abs(zscore(d["PC2"].astype(float), nan_policy="omit")) < 3) &
        (np.abs(zscore(d["hads_dep_total"].astype(float), nan_policy="omit")) < 3)
    ].copy()

    m_pc1 = smf.ols("PC1 ~ hads_dep_total + fo + age + C(gender)", data=d).fit(cov_type=robust)
    m_pc2 = smf.ols("PC2 ~ hads_dep_total + fo + age + C(gender)", data=d).fit(cov_type=robust)

    print(m_pc1.summary())
    print(m_pc2.summary())

    return d, m_pc1, m_pc2


def compute_prepost_change(df: pd.DataFrame, var: str) -> pd.DataFrame:
    """
    Compute Δvar = pre - post within each (patient, session).
    Returns long df with columns: patient, session, {var}_change
    """
    wide = (
        df.pivot_table(index=["patient", "session"], columns="tms", values=var)
          .reset_index()
    )
    if "pre" not in wide.columns or "post" not in wide.columns:
        raise ValueError(f"Need both pre and post for {var} change. Found {wide.columns.tolist()}")
    wide[f"{var}_change"] = wide["pre"].astype(float) - wide["post"].astype(float)
    return wide[["patient", "session", f"{var}_change"]]


def pc_change_predicts_next_symptoms(
    df: pd.DataFrame,
    symptom_col: str = "hads_dep_total",
    covariates: list[str] = ["age", "gender"],
    robust: str = "HC3",
):
    """
    Separate OLS for 1->2 and 2->3:
      s2 ~ PC2_change + s1 + age + gender
      s3 ~ PC2_change + s2 + age + gender
    """
    pc2_change = compute_prepost_change(df, "PC2").rename(columns={"PC2_change": "PC2_change"})

    sym_wide = (
        df.pivot_table(index="patient", columns="session", values=symptom_col)
          .reset_index()
          .rename(columns={1: "s1", 2: "s2", 3: "s3"})
    )

    # merge (pc2_change is long, sym_wide is wide)
    d = pc2_change.merge(sym_wide[["patient", "s1", "s2", "s3"]], on="patient", how="inner")

    # add covariates
    df_cov = df[["patient"] + covariates].drop_duplicates()
    d = d.merge(df_cov, on="patient", how="left")

    # essentials
    d = d.dropna(subset=["PC2_change", "s1", "s2", "s3"] + covariates).copy()
    d["session"] = d["session"].astype(int)

    # outlier removal on PC2_change only (keep simple)
    d = d[np.abs(zscore(d["PC2_change"], nan_policy="omit")) < 3].copy()

    # session 1->2
    d1 = d[d["session"] == 1].copy()
    d1["baseline_symptom"] = d1["s1"]
    m_s1 = smf.ols("s2 ~ PC2_change + baseline_symptom + age + C(gender)", data=d1).fit(cov_type=robust)
    print(m_s1.summary())

    # session 2->3
    d2 = d[d["session"] == 2].copy()
    d2["baseline_symptom"] = d2["s2"]
    m_s2 = smf.ols("s3 ~ PC2_change + baseline_symptom + age + C(gender)", data=d2).fit(cov_type=robust)
    print(m_s2.summary())

    # plot
    plot_symptom_change_correlation(d1, d2, m_s1, m_s2, fig_dir=fig_dir)

    return (d1, m_s1), (d2, m_s2)


# -----------------------------
# Plotting
# -----------------------------

def plot_transition_probs(transitions: np.ndarray, outpath: Path):
    trans_mean = np.mean(transitions, axis=(0, 1))
    n_states = trans_mean.shape[0]
    states = np.arange(1, n_states + 1)

    diag_vals = np.diag(trans_mean)
    mask = np.eye(n_states, dtype=bool)
    trans_masked = np.ma.masked_array(trans_mean, mask=mask)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5), constrained_layout=True)

    im = axes[0].imshow(trans_masked)
    axes[0].set_yticks(range(n_states))
    axes[0].set_xticks(range(n_states))
    axes[0].set_xticklabels(states)
    axes[0].set_yticklabels(states)
    axes[0].set_xlabel("To state")
    axes[0].set_ylabel("From state")
    fig.colorbar(im, ax=axes[0])

    axes[1].bar(states, diag_vals)
    axes[1].set_title("Self-transitions")
    axes[1].set_xlabel("State")
    axes[1].set_ylabel("Transition probability")
    axes[1].set_xticks(states)

    fig.savefig(outpath)
    plt.close(fig)


def plot_reg_prediction_style(
    ax,
    model,
    data,
    x_col,
    y_col,
    covariate_cols_for_pred,
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
        ax.text(-0.15, 1.1, panel_letter, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")


def plot_predicted_improvement(
    ax,
    model,
    data,
    x_col,
    y_next_col,
    baseline_col,
    xlabel,
    ylabel,
    panel_letter=None,
    n_grid=200,
):
    d = data.copy()
    d["improve_obs"] = d[baseline_col].astype(float) - d[y_next_col].astype(float)

    ax.scatter(d[x_col], d["improve_obs"], alpha=0.5, s=18, color="black")

    x_vals = np.linspace(float(d[x_col].min()), float(d[x_col].max()), n_grid)

    baseline_fixed = float(d[baseline_col].mean())
    pred_df = pd.DataFrame({
        x_col: x_vals,
        baseline_col: baseline_fixed,
        "age": float(d["age"].mean()),
        "gender": d["gender"].mode().iloc[0],
    })

    pred = model.get_prediction(pred_df)
    yhat_next = pred.predicted_mean
    ci_next = pred.conf_int(alpha=0.05)

    mean_improve = baseline_fixed - yhat_next
    low_improve = baseline_fixed - ci_next[:, 1]
    high_improve = baseline_fixed - ci_next[:, 0]

    ax.plot(x_vals, mean_improve, color="black", linewidth=3)
    ax.fill_between(x_vals, low_improve, high_improve, color="black", alpha=0.2)
    ax.axhline(0, linestyle=":", color="black", linewidth=1)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title("")

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    if panel_letter is not None:
        ax.text(-0.15, 1.1, panel_letter, transform=ax.transAxes, fontsize=20, fontweight="bold", va="top")


def plot_symptom_change_correlation(
    df_sess1, df_sess2,
    m_s1, m_s2,
    fig_dir: Path,
    out_name: str = "symptom_change_PC2change_predImprovement",
    figsize=(8, 3),
):
    fig, (ax1, ax2) = plt.subplots(
        1, 2,
        figsize=figsize,
        sharey=True,
        sharex=True,
        gridspec_kw={"wspace": 0.4}
    )

    plot_predicted_improvement(
        ax=ax1,
        model=m_s1,
        data=df_sess1,
        x_col="PC2_change",
        y_next_col="s2",
        baseline_col="baseline_symptom",
        xlabel="PC2 change (pre − post)",
        ylabel="Predicted symptom improvement",
        panel_letter="a",
    )

    plot_predicted_improvement(
        ax=ax2,
        model=m_s2,
        data=df_sess2,
        x_col="PC2_change",
        y_next_col="s3",
        baseline_col="baseline_symptom",
        xlabel="PC2 change (pre − post)",
        ylabel="Predicted symptom improvement",
        panel_letter="b",
    )

    fig.savefig(fig_dir / f"{out_name}.svg", dpi=300, bbox_inches="tight")
    fig.savefig(fig_dir / f"{out_name}.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

def plot_variance_explained(ax, pca, x_max=None, show_points=False):
    """
    Scree (per-PC %) + cumulative (%) from sklearn PCA.
    If x_max is set, only show first x_max PCs on the x-axis (but cumulative is still correct).
    """
    evr = np.asarray(pca.explained_variance_ratio_, dtype=float)  # fractions summing to 1
    cum = np.cumsum(evr) * 100.0
    per = evr * 100.0

    n = len(per)
    x = np.arange(1, n + 1)

    # Optionally restrict displayed range
    if x_max is not None:
        x = x[:x_max]
        per_plot = per[:x_max]
        cum_plot = cum[:x_max]
    else:
        per_plot = per
        cum_plot = cum

    ax.plot(x, per_plot, color="black", linewidth=2, label="Per-PC (%)")
    ax.plot(x, cum_plot, color="black", linewidth=3, linestyle="--", label="Cumulative (%)")

    if show_points:
        ax.scatter(x, per_plot, color="black", s=10)
        ax.scatter(x, cum_plot, color="black", s=10)

    ax.set_xlabel("Principal component")
    ax.set_ylabel("Variance explained (%)")
    ax.set_ylim(0, 100)

    # minimal axes
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

def plot_pc_loading_heatmap(ax, loading_vec, n_states, title=""):
    """
    Reshape off-diagonal PCA loadings (length n_states*(n_states-1))
    into a state x state matrix with diagonal masked.
    """
    mat = np.full((n_states, n_states), np.nan, dtype=float)
    mask = ~np.eye(n_states, dtype=bool)
    mat[mask] = loading_vec

    im = ax.imshow(mat, cmap="viridis")
    ax.set_xticks(range(n_states))
    ax.set_yticks(range(n_states))
    ax.set_xticklabels([str(i + 1) for i in range(n_states)])
    ax.set_yticklabels([str(i + 1) for i in range(n_states)])
    ax.set_xlabel("To state")
    ax.set_ylabel("From state")
    ax.set_title(title)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.set_ticks_position("left")
    ax.xaxis.set_ticks_position("bottom")

    return im


def make_supplementary_figure(transitions, pca, savepath_base, x_max=30):
    """
    Supplementary figure:
      a) mean transitions (off-diag heatmap + diagonal bars)
      b) variance explained (per-PC + cumulative)
    """
    import matplotlib.gridspec as gridspec

    fig = plt.figure(figsize=(11, 4))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1.2, 0.8, 1.2], wspace=0.6)

    axA = fig.add_subplot(gs[0, 0])
    axA_diag = fig.add_subplot(gs[0, 1])
    axB = fig.add_subplot(gs[0, 2])

    # panel a: transitions
    trans_mean = np.mean(transitions, axis=(0, 1))
    n_states_local = trans_mean.shape[0]
    states = np.arange(1, n_states_local + 1)

    mask = np.eye(n_states_local, dtype=bool)
    trans_masked = np.ma.masked_array(trans_mean, mask=mask)
    im = axA.imshow(trans_masked)
    axA.set_yticks(range(n_states_local))
    axA.set_xticks(range(n_states_local))
    axA.set_xticklabels(states)
    axA.set_yticklabels(states)
    axA.set_xlabel("To state")
    axA.set_ylabel("From state")

    cbar = fig.colorbar(im, ax=axA, fraction=0.046, pad=0.04)
    cbar.set_label("Transition probability")

    diag_vals = np.diag(trans_mean)
    axA_diag.bar(states, diag_vals, color="black")
    axA_diag.set_title("Self-transitions")
    axA_diag.set_xlabel("State")
    axA_diag.set_ylabel("Probability")
    axA_diag.set_xticks(states)

    for ax in (axA, axA_diag, axB):
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.yaxis.set_ticks_position("left")
        ax.xaxis.set_ticks_position("bottom")

    axA.text(-0.25, 1.1, "a", transform=axA.transAxes, fontsize=20, fontweight="bold", va="top")

    # panel b: variance explained
    plot_variance_explained(axB, pca, x_max=x_max)
    axB.text(-0.25, 1.1, "b", transform=axB.transAxes, fontsize=20, fontweight="bold", va="top")

    fig.savefig(savepath_base + ".svg", dpi=300, bbox_inches="tight")
    fig.savefig(savepath_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)


def make_main_figure(
    df_all,
    pca_out,
    n_states,
    symptom_col="hads_dep_total",
    robust="HC3",
    savepath_base="MAIN_pca_loadings_baseline_change",
):
    """
    Main figure:
      a) PC1 loadings heatmap
      b) PC2 loadings heatmap
      c) baseline PC1 ~ baseline HADS-D (prediction style)
      d) baseline PC2 ~ baseline HADS-D (prediction style)
      e) predicted symptom improvement 1->2 from ΔPC2
      f) predicted symptom improvement 2->3 from ΔPC2
    """
    import matplotlib.gridspec as gridspec

    # ---------- Baseline subset ----------
    d_base = df_all[(df_all["session"].astype(int) == 1) & (df_all["tms"].astype(str) == "pre")].copy()

    d_base = d_base[
        (np.abs(zscore(d_base["PC2"].astype(float), nan_policy="omit")) < 3) &
        (np.abs(zscore(d_base[symptom_col].astype(float), nan_policy="omit")) < 3)
    ].copy()

    # baseline models
    m_pc1 = smf.ols(f"PC1 ~ {symptom_col} + age + C(gender)", data=d_base).fit(cov_type=robust)
    m_pc2 = smf.ols(f"PC2 ~ {symptom_col} + age + C(gender)", data=d_base).fit(cov_type=robust)

    cov_pred = {
        "age": float(d_base["age"].mean()),
        "gender": d_base["gender"].mode().iloc[0],
    }

    # ---------- ΔPC2 and symptom transitions ----------
    pc2_change = compute_prepost_change(df_all, "PC2")  # patient, session, PC2_change

    sym_wide = (
        df_all.pivot_table(index="patient", columns="session", values=symptom_col)
              .reset_index()
              .rename(columns={1: "s1", 2: "s2", 3: "s3"})
    )

    d_change = pc2_change.merge(sym_wide[["patient", "s1", "s2", "s3"]], on="patient", how="inner")
    d_cov = df_all[["patient", "age", "gender"]].drop_duplicates()
    d_change = d_change.merge(d_cov, on="patient", how="left").dropna().copy()
    d_change["session"] = d_change["session"].astype(int)

    # outliers on PC2_change
    d_change = d_change[np.abs(zscore(d_change["PC2_change"], nan_policy="omit")) < 3].copy()

    # 1->2
    d1 = d_change[d_change["session"] == 1].copy()
    d1["baseline_symptom"] = d1["s1"]
    m_s1 = smf.ols("s2 ~ PC2_change + baseline_symptom + age + C(gender)", data=d1).fit(cov_type=robust)

    # 2->3
    d2 = d_change[d_change["session"] == 2].copy()
    d2["baseline_symptom"] = d2["s2"]
    m_s2 = smf.ols("s3 ~ PC2_change + baseline_symptom + age + C(gender)", data=d2).fit(cov_type=robust)

    # ---------- Layout ----------
    fig = plt.figure(figsize=(11, 10))
    gs = gridspec.GridSpec(3, 2, hspace=0.65, wspace=0.5)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])

    # ---------- a/b: loadings ----------
    loadings = pca_out["loadings"]  # (n_components, n_features_offdiag)
    im1 = plot_pc_loading_heatmap(axA, loadings[0, :], n_states, title="PC1 loadings")
    im2 = plot_pc_loading_heatmap(axB, loadings[1, :], n_states, title="PC2 loadings")

    cbar = fig.colorbar(im2, ax=[axA, axB], fraction=0.046, pad=0.02)
    cbar.set_label("Loading weight")

    axA.text(-0.25, 1.1, "a", transform=axA.transAxes, fontsize=20, fontweight="bold", va="top")
    axB.text(-0.25, 1.1, "b", transform=axB.transAxes, fontsize=20, fontweight="bold", va="top")

    # ---------- c/d: baseline regressions ----------
    plot_reg_prediction_style(
        ax=axC,
        model=m_pc1,
        data=d_base,
        x_col=symptom_col,
        y_col="PC1",
        covariate_cols_for_pred=cov_pred,
        xlabel="Baseline HADS-D",
        ylabel="Baseline PC1",
        panel_letter="c",
    )

    plot_reg_prediction_style(
        ax=axD,
        model=m_pc2,
        data=d_base,
        x_col=symptom_col,
        y_col="PC2",
        covariate_cols_for_pred=cov_pred,
        xlabel="Baseline HADS-D",
        ylabel="Baseline PC2",
        panel_letter="d",
    )

    # ---------- e/f: predicted improvement ----------
    plot_predicted_improvement(
        ax=axE,
        model=m_s1,
        data=d1,
        x_col="PC2_change",
        y_next_col="s2",
        baseline_col="baseline_symptom",
        xlabel="ΔPC2 (pre − post)",
        ylabel="Predicted symptom improvement",
        panel_letter="e",
    )

    plot_predicted_improvement(
        ax=axF,
        model=m_s2,
        data=d2,
        x_col="PC2_change",
        y_next_col="s3",
        baseline_col="baseline_symptom",
        xlabel="ΔPC2 (pre − post)",
        ylabel="Predicted symptom improvement",
        panel_letter="f",
    )

    fig.savefig(savepath_base + ".svg", dpi=300, bbox_inches="tight")
    fig.savefig(savepath_base + ".png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    return {
        "baseline_models": (m_pc1, m_pc2),
        "change_models": (m_s1, m_s2),
        "baseline_df": d_base,
        "change_dfs": (d1, d2),
    }


def save_pca_results(pca_out, filepath):
    joblib.dump(pca_out, filepath)
    print(f"Saved PCA results to {filepath}")

if __name__ == "__main__":

    n_sessions = 6
    all_states = [10]

    for n_states in all_states:

        df_all, pca_out = add_pca_to_clinical(hmm_dir, n_states=n_states, n_sessions=n_sessions)

        # Supplementary: transitions + variance explained
        make_supplementary_figure(
            transitions=pca_out["transitions"],
            pca=pca_out["pca"],
            savepath_base=str(fig_dir / f"SUPP_transitions_variance_states{n_states}"),
            x_max=30,  # increase/decrease if you want
        )

        # Main: loadings + baseline regressions + symptom improvement predictions
        make_main_figure(
            df_all=df_all,
            pca_out=pca_out,
            n_states=n_states,
            symptom_col="hads_dep_total",
            robust="HC3",
            savepath_base=str(fig_dir / f"MAIN_pca_loadings_baseline_improvement_states{n_states}"),
        )


        # (Optional) still print model summaries in console:
        d_baseline, m_pc1, m_pc2 = baseline_pc_vs_hads(df_all, robust="HC3")
        pc_change_predicts_next_symptoms(df_all, symptom_col="hads_dep_total", covariates=["age", "gender"], robust="HC3")