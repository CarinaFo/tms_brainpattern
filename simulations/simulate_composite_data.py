import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# ---------- Simulation + PCA figure (reviewer-friendly) ----------
np.random.seed(1)

# 10 patients, 3 FO states that sum to 1
# Two "latent patterns" (groups) so PCA has meaningful structure to recover
alpha_group1 = [6, 2, 1]   # more State1
alpha_group2 = [1, 5, 2]   # more State2

FO = np.vstack([
    np.random.dirichlet(alpha_group1, size=5),
    np.random.dirichlet(alpha_group2, size=5)
])

patients = [f"P{i+1}" for i in range(10)]
groups = np.array(["Group 1"]*5 + ["Group 2"]*5)

df = pd.DataFrame(FO, columns=["State1", "State2", "State3"])
df["patient"] = patients
df["group"] = groups

# Sanity check: each row sums to 1
row_sums = df[["State1", "State2", "State3"]].sum(axis=1)


# --- Plot original FO space ---
plt.figure()

# Scatter (State1 vs State2; State3 implicit)
plt.scatter(df["State1"], df["State2"])

# Label points
for pid in df.index:
    plt.text(
        df.loc[pid, "State1"],
        df.loc[pid, "State2"],
        pid,
        fontsize=9
    )

plt.xlabel("State 1 fractional occupancy")
plt.ylabel("State 2 fractional occupancy")
plt.title("Original fractional occupancy space\n(State3 = 1 − State1 − State2)")
plt.show()

# PCA on raw FOs (no standardisation)
X = df[["State1", "State2", "State3"]].values
pca2 = PCA(n_components=2)
scores2 = pca2.fit_transform(X)

# Fit full PCA to show rank deficiency (3rd variance ~0)
pca3 = PCA(n_components=3).fit(X)
scores3 = pca3.fit_transform(X)
evr = pca2.explained_variance_ratio_
ev_full = pca3.explained_variance_

# ---------- Plot ----------
plt.figure()

# Plot each group separately (default matplotlib color cycle)
for g in np.unique(groups):
    idx = (groups == g)
    plt.scatter(scores3[idx, 0], scores3[idx, 1], label=g)

# Label points
for i in df.index:
    plt.text(scores3[i, 0], scores3[i, 1], i, fontsize=9)

plt.xlabel(f"PC1 ({evr[0]*100:.1f}% var)")
plt.ylabel(f"PC2 ({evr[1]*100:.1f}% var)")
plt.title("PCA on Simulated Fractional Occupancies (3 states; rows sum to 1)")
plt.legend(frameon=False)
plt.show()

# ---------- Diagnostics for reviewer response ----------
print("Row sums (should all be 1):", np.round(row_sums.values, 6))
print("Explained variance ratio (PC1, PC2):", np.round(evr, 6))
print("Explained variance (3 PCs; 3rd should be ~0):", np.round(ev_full, 12))
