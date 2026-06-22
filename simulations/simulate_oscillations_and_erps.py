import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# Simulation settings
# -----------------------------

sfreq = 1000  # Hz
tmin = -0.5   # seconds
tmax = 0    # seconds

times = np.arange(tmin, tmax, 1 / sfreq)

# -----------------------------
# Alpha oscillation
# -----------------------------

alpha_freq = 10  # Hz
alpha_amp = 1    # microvolts

alpha = alpha_amp * np.sin(2 * np.pi * alpha_freq * times)

# -----------------------------
# Evoked components
# -----------------------------

def gaussian_peak(times, peak_time, amplitude, width):
    """
    Create a Gaussian-shaped ERP component.

    peak_time in seconds
    amplitude in microvolts
    width is the standard deviation in seconds
    """
    return amplitude * np.exp(-0.5 * ((times - peak_time) / width) ** 2)


p50 = gaussian_peak(times, peak_time=0.050, amplitude=3, width=0.015)
n100 = gaussian_peak(times, peak_time=0.100, amplitude=-5, width=0.025)
p300 = gaussian_peak(times, peak_time=0.300, amplitude=8, width=0.060)

evoked = p50 + n100 + p300

# -----------------------------
# Noise
# -----------------------------

noise_amp = 1.5
noise = noise_amp * np.random.randn(len(times))

# -----------------------------
# Final simulated EEG signal
# -----------------------------

signal = alpha + evoked + noise

# -----------------------------
# Plot
# -----------------------------

plt.figure(figsize=(10, 5))

#plt.plot(times * 1000, signal, label="Simulated EEG")
plt.plot(times * 1000, alpha, linestyle="--", label="Alpha oscillation")
#plt.plot(times * 1000, evoked, linewidth=2, label="Evoked response")

#plt.axvline(0, linestyle=":", label="Stimulus onset")


plt.xlabel("Time (ms)")
plt.ylabel("Amplitude (µV)")
#plt.title("Simulated single-channel EEG: alpha + P50/N100/P300 evoked response")
plt.legend()
plt.tight_layout()
plt.savefig('simulated_alpha_erp.svg')
plt.show()