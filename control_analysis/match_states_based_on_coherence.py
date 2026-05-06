import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path
from scipy.stats import pearsonr
from scipy.optimize import linear_sum_assignment


base_dir = Path("L:/Lab_LucaC/Carina")
hmm_dir = base_dir / "canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered"
fig_dir = hmm_dir / "figures/state_matching_coherence"
fig_dir.mkdir(parents=True, exist_ok=True)

ref_states = 10
compare_states = [6, 8, 12]
sessions = range(6)


def load_group_coherence_vectors(n_states, session):
    """
    Input coh shape:
    subjects x states x parcels x parcels x frequencies

    Returns:
    states x edges
    """

    coh = np.load(hmm_dir / f"coh_{session}_{n_states}.npy")
    w = np.load(hmm_dir / f"w_{session}_{n_states}.npy")

    # group average across subjects
    # result: states x parcels x parcels x frequencies
    gcoh = np.average(coh, axis=0, weights=w)

    # average across frequencies
    # result: states x parcels x parcels
    gcoh_mean = gcoh.mean(axis=-1)

    # vectorise upper triangle
    n_parcels = gcoh_mean.shape[1]
    triu = np.triu_indices(n_parcels, k=1)

    vectors = np.array([
        gcoh_mean[state][triu]
        for state in range(gcoh_mean.shape[0])
    ])

    return vectors

def order_states_by_mean_coherence(coh_vectors):
    """
    High mean coherence first.
    """
    mean_coh = coh_vectors.mean(axis=1)
    order = np.argsort(mean_coh)[::-1]
    return order


def correlation_matrix(ref_vectors, other_vectors):
    """
    Rows = other states
    Columns = reference states
    """

    sim = np.zeros((other_vectors.shape[0], ref_vectors.shape[0]))

    for i in range(other_vectors.shape[0]):
        for j in range(ref_vectors.shape[0]):
            sim[i, j] = pearsonr(other_vectors[i], ref_vectors[j])[0]

    return sim


def match_states_to_reference(sim):
    """
    Match other states to reference states by maximising correlation.
    """

    rows, cols = linear_sum_assignment(-sim)
    return rows, cols


all_matches = []

for session in sessions:
    print(f"Session {session}")

    # load and order reference 10-state solution
    ref_vectors_raw = load_group_coherence_vectors(ref_states, session)
    ref_order = order_states_by_mean_coherence(ref_vectors_raw)

    #ref_vectors = ref_vectors_raw[ref_order]
    # Here we use the original ordering
    ref_vectors = ref_vectors_raw

    for ns in compare_states:
        print(f"  Matching {ns} states to {ref_states} states")

        other_vectors = load_group_coherence_vectors(ns, session)

        sim = correlation_matrix(ref_vectors, other_vectors)
        rows, cols = match_states_to_reference(sim)

        # rows = states in other model
        # cols = ordered reference states
        matched_r = sim[rows, cols]

        for other_idx, ref_idx, r in zip(rows, cols, matched_r):
            all_matches.append({
                "session": session,
                "comparison": f"{ns} vs {ref_states}",
                "n_states": ns,
                "other_state_original": other_idx + 1,
                "ref_state_ordered": ref_idx + 1,
                "ref_state_original": ref_order[ref_idx] + 1,
                "spatial_r": r,
            })

        # heatmap
        plt.figure(figsize=(7, 4))
        sns.heatmap(
            sim,
            cmap="coolwarm",
            vmin=-1,
            vmax=1,
            center=0,
            annot=True,
            fmt=".2f",
            cbar_kws={"label": "Coherence correlation"},
        )
        plt.xlabel("10-state reference, ordered high → low coherence")
        plt.ylabel(f"{ns}-state solution")
        plt.title(f"Session {session}: {ns} vs {ref_states}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"coh_match_heatmap_s{session}_{ns}_vs_{ref_states}.svg")
        plt.savefig(fig_dir / f"coh_match_heatmap_s{session}_{ns}_vs_{ref_states}.png", dpi=300)
        plt.close()


matches = pd.DataFrame(all_matches)
matches.to_csv(fig_dir / "coherence_state_matches.csv", index=False)


# summary plot
summary = (
    matches
    .groupby(["session", "comparison"], as_index=False)["spatial_r"]
    .mean()
)

plt.figure(figsize=(6, 4))

sns.stripplot(
    data=summary,
    x="comparison",
    y="spatial_r",
    color="black",
    jitter=0.15,
    size=6,
)

sns.pointplot(
    data=summary,
    x="comparison",
    y="spatial_r",
    color="black",
    errorbar=("ci", 95),
    join=False,
    capsize=0.15,
)

plt.axhline(0, linestyle=":", color="black", linewidth=1)
plt.ylabel("Mean matched coherence correlation")
plt.xlabel("HMM resolution comparison")
plt.ylim(-1, 1)
plt.tight_layout()

plt.savefig(fig_dir / "coherence_state_matching_summary.svg")
plt.savefig(fig_dir / "coherence_state_matching_summary.png", dpi=300)
plt.close()

print(matches)


ref_state = 2  # original 10-state ordering
ref_idx = ref_state - 1

state2_matches = []

for session in sessions:
    print(f"Session {session}")

    ref_vectors = load_group_coherence_vectors(ref_states, session)
    ref_state2_vec = ref_vectors[ref_idx]

    for ns in compare_states:
        print(f"  Comparing {ns} states to 10-state state {ref_state}")

        other_vectors = load_group_coherence_vectors(ns, session)

        # correlation of every state in other solution with 10-state state 2
        corrs = np.array([
            pearsonr(other_vectors[i], ref_state2_vec)[0]
            for i in range(other_vectors.shape[0])
        ])

        best_idx = np.argmax(corrs)

        for i, r in enumerate(corrs):
            state2_matches.append({
                "session": session,
                "n_states": ns,
                "comparison": f"{ns} vs 10",
                "other_state": i + 1,
                "ref_state": ref_state,
                "spatial_r": r,
                "is_best_match": i == best_idx,
            })

state2_matches = pd.DataFrame(state2_matches)
state2_matches.to_csv(fig_dir / "state2_10state_coherence_matches.csv", index=False)