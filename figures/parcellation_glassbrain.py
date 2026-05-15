import nibabel as nib
import numpy as np
import seaborn as sns
from nilearn import plotting
from matplotlib.colors import ListedColormap

# Load Giles parcellation
img4d = nib.load(r"L:\Lab_LucaC\Carina\canonical_hmm_finalsample\atlas-Giles_nparc-38_space-MNI_res-8x8x8.nii.gz")
data4d = img4d.get_fdata()  # shape: x, y, z, 38

# Convert 4D -> 3D labels
labels = np.argmax(data4d, axis=3) + 1
labels[np.max(data4d, axis=3) <= 0] = 0

label_img = nib.Nifti1Image(
    labels.astype(np.int16),
    img4d.affine,
    img4d.header
)

# Generate 38 clearly distinct colours
# Soft pastel palette
colors = sns.husl_palette(
    20,
    s=0.45,   # saturation
    l=0.72    # lightness
)

cmap = ListedColormap(colors)

# Plot left lateral glass brain
display = plotting.plot_glass_brain(
    label_img,
    display_mode="l",
    cmap=cmap,
    colorbar=False,
    black_bg=False,
    threshold=0.5,
    alpha=0.55,
    plot_abs=False,
)

display.savefig(
    "giles38_glassbrain_left.svg",
    dpi=300,
    transparent=True,
)

display.close()