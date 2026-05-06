"""
Match HMM states across resolutions using spatial correlations of group-average power maps.

Reference model: 10 states
Comparisons: 6 vs 10, 8 vs 10, 12 vs 10
Done separately for each EEG session.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment # find best matching state

from osl_dynamics.analysis import power


# -----------------------------
# Settings
# -----------------------------
base_dir = Path("L:/Lab_LucaC/Carina")
hmm_dir = base_dir / "canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered"
fig_dir = hmm_dir / "figures/state_matching"
fig_dir.mkdir(parents=True, exist_ok=True)

ref_states = 10
compare_states = [8]
sessions = range(6)

# -----------------------------
# Helper functions
# -----------------------------
def load_group_power_maps(n_states, session, subtract_mean=True):
    """
    Load spectra and compute group-average state power maps.

    Returns
    -------
    p : array, shape = n_states x n_parcels
        Group-average power map per state.
    """

    f = np.load(hmm_dir / f"f_{session}_{n_states}.npy")
    psd = np.load(hmm_dir / f"psd_{session}_{n_states}.npy")
    w = np.load(hmm_dir / f"w_{session}_{n_states}.npy")
    fo = np.load(hmm_dir / f"fo_{session}_{n_states}.npy")
    wb_comp = np.load(hmm_dir / f"nnmf_{session}_{n_states}.npy")

    # group-average spectra across subjects using subject weights
    gpsd = np.average(psd, axis=0, weights=w)

    # group-average FO across subjects
    gfo = np.average(fo, axis=0, weights=w)

    # power maps per state
    p = power.variance_from_spectra(f, gpsd, components=wb_comp)

    # get the first wideband component (NNMF)
    po = p[0,:,:]

    # match what you plot with subtract_mean=True
    if subtract_mean:
        po = po - np.average(po, axis=0, weights=gfo)

    return po # shape: brain states, parcels


def spatial_corr_matrix(maps_a, maps_b):
    """
    Compute state-by-state spatial Pearson correlations.

    maps_a: n_states_a x n_features
    maps_b: n_states_b x n_features
    """

    maps_a = np.asarray(maps_a)
    maps_b = np.asarray(maps_b)

    sim = np.zeros((maps_a.shape[0], maps_b.shape[0]))

    for i in range(maps_a.shape[0]):
        for j in range(maps_b.shape[0]):
            x = maps_a[i].ravel()
            y = maps_b[j].ravel()

            valid = np.isfinite(x) & np.isfinite(y)

            if valid.sum() < 3:
                sim[i, j] = np.nan
            else:
                sim[i, j] = pearsonr(x[valid], y[valid])[0]

    return sim


def match_to_reference(sim):
    """
    Hungarian matching to maximise spatial correlation.
    """

    cost = -sim.copy()
    cost[~np.isfinite(cost)] = 1e6

    row_ind, col_ind = linear_sum_assignment(cost)

    return row_ind, col_ind


# -----------------------------
# Main analysis
# -----------------------------
all_rows = []

for session in sessions:
    print(f"Session {session}")

    ref_maps = load_group_power_maps(ref_states, session)

    for ns in compare_states:
        print(f"  Matching {ns} vs {ref_states}")

        maps = load_group_power_maps(ns, session)

        sim = spatial_corr_matrix(maps, ref_maps)

        rows, cols = match_to_reference(sim)

        # save heatmap
        plt.figure(figsize=(6, 4))
        sns.heatmap(
            sim,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            cbar_kws={"label": "Spatial correlation"},
        )
        plt.xlabel(f"{ref_states}-state model")
        plt.ylabel(f"{ns}-state model")
        plt.title(f"Session {session}: {ns} vs {ref_states}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"spatial_corr_heatmap_s{session}_{ns}_vs_{ref_states}.svg")
        plt.savefig(fig_dir / f"spatial_corr_heatmap_s{session}_{ns}_vs_{ref_states}.png", dpi=300)
        plt.show()

        for alt_state, ref_state in zip(rows, cols):
            all_rows.append({
                "session": session,
                "comparison": f"{ns} vs {ref_states}",
                "n_states": ns,
                "alt_state": alt_state + 1,
                "ref_state": ref_state + 1,
                "r": sim[alt_state, ref_state],
            })


results = pd.DataFrame(all_rows)
results.to_csv(fig_dir / "state_matching_spatial_correlations.csv", index=False)


# -----------------------------
# Summary plot
# -----------------------------
summary = (
    results
    .groupby(["session", "comparison"], as_index=False)["r"]
    .mean()
)

plt.figure(figsize=(6, 4))

sns.stripplot(
    data=summary,
    x="comparison",
    y="r",
    color="black",
    size=6,
    jitter=0.15,
)

sns.pointplot(
    data=summary,
    x="comparison",
    y="r",
    color="black",
    errorbar=("ci", 95),
    join=False,
    capsize=0.15,
)

plt.axhline(0, linestyle=":", color="black", linewidth=1)
plt.ylabel("Mean matched spatial correlation")
plt.xlabel("HMM resolution comparison")
plt.ylim(-1, 1)
plt.tight_layout()

plt.savefig(fig_dir / "state_matching_summary.svg")
plt.savefig(fig_dir / "state_matching_summary.png", dpi=300)
plt.close()

print(results)
print(summary)