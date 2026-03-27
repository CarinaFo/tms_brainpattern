import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

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
# ----------------------------
# Simulate EEG-like signals
# ----------------------------
rng = np.random.default_rng(7)

fs = 250
duration = 10
t = np.arange(0, duration, 1 / fs)
n = len(t)
n_channels = 3

signals = []
for ch in range(n_channels):
    sig = (
        0.9 * np.sin(2 * np.pi * (8 + 0.3 * ch) * t + rng.uniform(0, 2*np.pi)) +
        0.4 * np.sin(2 * np.pi * (18 + ch) * t + rng.uniform(0, 2*np.pi)) +
        0.6 * np.sin(2 * np.pi * 1.2 * t + rng.uniform(0, 2*np.pi)) +
        0.35 * rng.normal(size=n)
    )
    signals.append(sig)

signals = np.array(signals)

# Normalize for clean overlay
signals = (signals - signals.mean(axis=1, keepdims=True)) / signals.std(axis=1, keepdims=True)

# ----------------------------
# Single active state
# ----------------------------
state_start = 3.2
state_end = 5.4
state_label = "State 2"
state_color = "#FFD700"  # gold

# ----------------------------
# Plot
# ----------------------------
plt.figure(figsize=(12, 5))

# Highlight state window
plt.axvspan(state_start, state_end, color=state_color, alpha=0.25, zorder=0)

# Plot all signals in black
for i in range(n_channels):
    plt.plot(t, signals[i], color="black", lw=1.2, alpha=0.8)

# Mark boundaries
plt.axvline(state_start, color=state_color, ls="--", lw=1.5)
plt.axvline(state_end, color=state_color, ls="--", lw=1.5)

# Label
ymax = np.max(signals)
plt.text(
    (state_start + state_end) / 2,
    ymax + 0.3,
    state_label,
    ha="center",
    va="bottom",
    fontsize=11,
    fontweight="bold"
)


plt.grid(alpha=0.2)

plt.tight_layout()
plt.savefig(f'{fig_dir}/stc_example.svg')
plt.show()