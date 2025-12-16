# Plot figure 1 (symptom improvement in HADS score over 3 sessions)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
import ptitprince as pt
import statsmodels.formula.api as smf
import os
from scipy.stats import zscore

# --------------------------------------------------
# Configuration
# --------------------------------------------------
home_dir = Path("L:/Lab_LucaC/Carina/")
n_states = 8

hmm_dir = Path(f"{home_dir}/prepared_data_80patients_giles_newmodel")
csv_path = Path(f"{hmm_dir}/hmm_demo_quest_{n_states}.csv")
fig_dir = Path(f'{hmm_dir}/figures')

if not os.path.exists(fig_dir):
    os.makedirs(fig_dir)

# plot figure 1 (symptom change, coefficients)
#plot_symptom_change_hads()
#plot_regression_coeffs_demo()

# --------------------------------------------------
# Data loading & preprocessing
# --------------------------------------------------
def load_and_prep_data(csv_path, n_states=n_states, exclude_repeater: bool = True):
    
    # read csv file containing clinical and hmm data
    df = pd.read_csv(csv_path)

    # exclude repeater IDs and very noisy IDs
    exclude_ids = ["144R", "127", "182"]
    df = df[~df["patient"].isin(exclude_ids)]
    if exclude_repeater:
        df = df[~df["patient"].str.contains("R")]

    print(f"Analyzing {df['patient'].nunique()} patients")

    df["state"] = df["state"] + 1  # we want states starting from 1

    # propagate demographic vars
    for col in ["age", "gender", "responder", "group"]:
        df[col] = df.groupby("patient")[col].transform("first")

    # cast common categorical columns (optional, but safe)
    for col in ["patient", "session", "tms", "state", "group", "responder"]:
        df[col] = df[col].astype("category", errors="ignore")
    
    df['years_with_depression'] = df['age'] - df['age_of_symptom_onset']

    return df


def plot_symptom_change_hads():

    df = load_and_prep_data(csv_path)

    # Group by patient and session, then get the dep_hads score
    dep_hads_per_session = df.groupby(['patient', 'session'])['dep_hads'].first().reset_index()

    # Plot style
    sns.set_theme(style="whitegrid")
    plt.rcParams['font.family'] = 'Arial'

    plt.figure(figsize=(10,6))

    # Create Raincloud plot
    ax = pt.RainCloud(
        x='session',
        y='dep_hads',
        data=dep_hads_per_session,
        bw=.25,              # bandwidth of kernel density
        width_viol=.6,      # width of the violin part
        move=.25,
        alpha=0.9,           # offset
        orient='v'          # vertical orientation
    )

    ax.yaxis.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)
    ax.set_xlabel("Treatment timepoint", fontsize=22, labelpad=10)
    ax.set_ylabel("HADS-D", fontsize=22, labelpad=10)
    ax.set_xticklabels(["Pre", "Mid", "Post"], fontsize=18)

    ax.set_yticks([0, 10, 20])
    ax.set_yticklabels([0, 10, 20], fontsize=18)  # set labels and font size
    ax.set_ylim(-1, 22)  # a bit of padding above 20 for aesthetics

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    sns.despine()

    # -------------------------
    # Save high-quality figure
    # -------------------------
    plt.tight_layout()
    plt.savefig(f"{fig_dir}/raincloud_dep_hads.svg", dpi=300, bbox_inches='tight')
    plt.savefig(f"{fig_dir}/raincloud_dep_hads.png", dpi=300, bbox_inches='tight')
    plt.show()


def plot_regression_coeffs_demo():

    df = load_and_prep_data(csv_path)

    # Group by patient and session, then get the dep_hads score
    dep_hads_per_session = df.groupby(['patient', 'session'])['dep_hads'].first().reset_index()

    # Add static patient info (assuming one row per patient)
    demo = df[["patient", "age", "gender", "group", "years_with_depression"]].drop_duplicates("patient")
    df_session = dep_hads_per_session.merge(demo, on="patient", how="left")
    df_session = df_session.dropna(subset=["years_with_depression"])

    df_s1 = df_session[df_session["session"] == 1].copy()

    df_s1["age_z"] = zscore(df_s1["age"])
    df_s1["years_with_depression_z"] = zscore(df_s1["years_with_depression"])

    model = smf.ols("dep_hads ~ age_z + gender + group + years_with_depression_z", data=df_s1).fit()
    print(model.summary())

    # Pivot to have Session 1 and Session 3 HADS in separate columns
    df_wide = df_session.pivot(index="patient", columns="session", values="dep_hads").reset_index()
    df_wide = df_wide.rename(columns={1: "HADS_1", 3: "HADS_3"})

    # Compute improvement (Session 1 - Session 3)
    df_wide["HADS_improvement"] = df_wide["HADS_1"] - df_wide["HADS_3"]

    # Merge static patient info
    df_wide = df_wide.merge(demo, on="patient", how="left")

    df_wide["age_z"] = zscore(df_wide["age"])
    df_wide["years_with_depression_z"] = zscore(df_wide["years_with_depression"])

    model = smf.ols(
        "HADS_improvement ~ age_z + gender + group + years_with_depression_z",
        data=df_wide
    ).fit()

    print(model.summary())

    # plot the coefficients
    coefs = model.params
    conf = model.conf_int()
    conf.columns = ['lower', 'upper']

    coefs_df = conf.copy()
    coefs_df['coef'] = coefs

    # Drop intercept
    coefs_df = coefs_df.drop('Intercept').reset_index().rename(columns={'index':'Predictor'})

    # Create a mapping from model variable names → friendly labels
    rename_dict = {
        "age_z": "Age (z-scored)",
        "years_with_depression_z": "Years with Depression (z-scored)",
        "gender[T.Male]": "Gender: Male vs Female",
        "group[T.2.0]": "Research Tier: Natural vs RCT",
        "group[T.3.0]": "Research Tier: Neurological vs RCT"
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