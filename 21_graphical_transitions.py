import joblib
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch

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

n_states=10

hmm_dir = base_dir / "Lab_LucaC/Carina/canonical_hmm_finalsample/hmm_fits_05Hzcanonical_1Hzfiltered"
fig_dir = hmm_dir / "figures"
fig_dir.mkdir(parents=True, exist_ok=True)

pca_path = hmm_dir / f"pca_results_{n_states}states.joblib"

def offdiag_vector_to_matrix(vec, n_states):
    """
    Reconstruct n_states x n_states matrix from an off-diagonal vector.

    vec length must be n_states * (n_states - 1).
    Diagonal is filled with np.nan.
    """
    vec = np.asarray(vec, dtype=float)
    expected = n_states * (n_states - 1)
    if vec.size != expected:
        raise ValueError(f"Expected vector of length {expected}, got {vec.size}")

    mat = np.full((n_states, n_states), np.nan, dtype=float)
    mask = ~np.eye(n_states, dtype=bool)
    mat[mask] = vec
    return mat


def circular_positions(n_states, radius=2.0):
    """
    Put state 1 at the top, then clockwise.
    """
    angles = np.linspace(np.pi / 2, np.pi / 2 - 2 * np.pi, n_states, endpoint=False)
    return {
        i: (radius * np.cos(a), radius * np.sin(a))
        for i, a in enumerate(angles)
    }


def get_top_loading_edges(loading_vec, n_states, top_frac=0.20):
    """
    Return top edges by absolute loading magnitude.
    Each edge is (from_state, to_state, loading, abs_loading).
    """
    mat = offdiag_vector_to_matrix(loading_vec, n_states)

    edges = []
    for i in range(n_states):
        for j in range(n_states):
            if i == j:
                continue
            w = mat[i, j]
            if np.isnan(w):
                continue
            edges.append((i, j, float(w), abs(float(w))))

    n_keep = max(1, int(np.ceil(top_frac * len(edges))))
    edges = sorted(edges, key=lambda x: x[3], reverse=True)
    return edges[:n_keep]


def plot_pc_loading_graph(
    ax,
    loading_vec,
    n_states,
    top_frac=0.20,
    title="",
    node_radius=0.18,
    layout_radius=2.0,
    show_labels=True,
):
    """
    Plot top PCA loading transitions as a directed graph.

    Conventions:
    - thicker edge = stronger absolute loading
    - solid black = positive loading
    - dashed gray = negative loading
    """
    edges = get_top_loading_edges(loading_vec, n_states=n_states, top_frac=top_frac)
    pos = circular_positions(n_states, radius=layout_radius)

    # draw nodes
    for i, (x, y) in pos.items():
        circ = Circle((x, y), node_radius, facecolor="white", edgecolor="black", lw=1.8, zorder=3)
        ax.add_patch(circ)
        ax.text(x, y, str(i + 1), ha="center", va="center", fontsize=11, zorder=4)

    if len(edges) == 0:
        ax.axis("off")
        ax.set_title(title)
        return

    abs_vals = np.array([e[3] for e in edges], dtype=float)
    amin, amax = abs_vals.min(), abs_vals.max()

    def scale_width(v):
        if amax == amin:
            return 2.5
        return 1.0 + 4.5 * (v - amin) / (amax - amin)

    # draw weakest first, strongest last
    for i, j, w, aw in sorted(edges, key=lambda x: x[3]):
        x1, y1 = pos[i]
        x2, y2 = pos[j]

        # separate bidirectional arrows slightly
        rad = 0.14 if i < j else -0.14

        color = "black" if w >= 0 else "0.45"
        linestyle = "-" if w >= 0 else "--"

        arrow = FancyArrowPatch(
            (x1, y1),
            (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>",
            mutation_scale=11,
            shrinkA=16,
            shrinkB=16,
            lw=scale_width(aw),
            linestyle=linestyle,
            color=color,
            alpha=0.9,
            zorder=2,
        )
        ax.add_patch(arrow)

        if show_labels:
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            dy = 0.08 if rad > 0 else -0.08
            ax.text(mx, my + dy, f"{w:.2f}", fontsize=8, ha="center", va="center")

    ax.set_aspect("equal")
    ax.set_xlim(-2.6, 2.6)
    ax.set_ylim(-2.6, 2.6)
    ax.axis("off")
    ax.set_title(title)

    ax.text(
        0.5, -0.08,
        "Top 20% |loadings|  •  solid = positive  •  dashed = negative",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=9,
    )


def plot_pc1_pc2_loading_graphs(loadings, n_states=10, top_frac=0.20, figsize=(12, 6)):
    """
    loadings: array of shape (n_components, n_features)
              e.g. sklearn PCA.components_

    For 10 states, n_features should be 90.
    """
    loadings = np.asarray(loadings)

    expected_features = n_states * (n_states - 1)
    if loadings.ndim != 2:
        raise ValueError("loadings must be 2D, e.g. shape (n_components, n_features)")
    if loadings.shape[1] != expected_features:
        raise ValueError(
            f"Expected {expected_features} features for {n_states} states, "
            f"but got loadings.shape = {loadings.shape}"
        )

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    plot_pc_loading_graph(
        axes[0],
        loading_vec=loadings[0, :],   # PC1
        n_states=n_states,
        top_frac=top_frac,
        title="PC1 loading graph",
        show_labels=False,
    )

    plot_pc_loading_graph(
        axes[1],
        loading_vec=loadings[1, :],   # PC2
        n_states=n_states,
        top_frac=top_frac,
        title="PC2 loading graph",
        show_labels=False,
    )

    plt.tight_layout()
    plt.savefig(f"{fig_dir}/pc1_pc2_loading_graphs.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{fig_dir}/pc1_pc2_loading_graphs.svg", dpi=300, bbox_inches="tight")
    plt.show()

def load_pca_results(filepath):
    pca_out = joblib.load(filepath)
    return pca_out

# load PCA loadings on transitions
pca = load_pca_results(pca_path)

plot_pc1_pc2_loading_graphs(
    pca.components_, n_states=10, top_frac=0.10
)
