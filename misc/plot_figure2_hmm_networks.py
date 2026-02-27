# plot figure 2 (HMM networks)
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.image as mpimg
from pathlib import Path

# setting for nature publishing
plt.rcParams['pdf.fonttype']=42

# linux doesn't have Arial
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 14,
    "axes.labelsize": 18,
    "axes.titlesize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

system='linux'

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/canonical_hmm_finalsample')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:/Lab_LucaC/Carina/")
else:
    "No available system path defined *windows* or *linux*"

save_dir = Path(f'{base_dir}/hmm_fits_05Hzcanonical_1Hzfiltered')

state=10
ses=0

save_dir_plots = f'{save_dir}/plots_{state}'
n_states = 10
states_per_row=5

fig = plt.figure(figsize=(18, 12))
gs = gridspec.GridSpec(
    nrows=7, # add a row for more space between states
    ncols=5,
    height_ratios=[1, 1, 1, 0.25, 1, 1, 1],
    hspace=0.05,
    wspace=0.05
)

for i in range(n_states):
    state_row = i // states_per_row      # 0 or 1
    state_col = i % states_per_row       # 0–4

    # Grid positions
    col_start = state_col * 1
    row_start = state_row * 3

    if state_row == 0:
        row_start = 0
    else:
        row_start = 4   # skip spacer row

    # load images
    pow_img = mpimg.imread(f"{save_dir_plots}/pow_{ses}_{i:02d}.png")
    psd_img = mpimg.imread(f"{save_dir_plots}/psd_{i:02d}_{ses}.png")
    coh_img = mpimg.imread(f"{save_dir_plots}/coh_{ses}_{i:02d}.png")

    # --- Power map (top, spanning two columns) ---
    ax_pow = fig.add_subplot(gs[row_start, col_start])
    ax_pow.imshow(pow_img)
    ax_pow.axis("off")
    ax_pow.set_title(f"State {i+1}", fontsize=18)

    # --- PSD (bottom left) ---
    ax_psd = fig.add_subplot(gs[row_start + 1, col_start])
    ax_psd.imshow(psd_img)
    ax_psd.axis("off")

    # --- Coherence (bottom right) ---
    ax_coh = fig.add_subplot(gs[row_start + 2, col_start])
    ax_coh.imshow(coh_img)
    ax_coh.axis("off")

plt.tight_layout()
plt.savefig(f"{save_dir_plots}/figure_states_overview.png", dpi=300)
plt.savefig(f"{save_dir_plots}/figure_states_overview.svg")
plt.show()
