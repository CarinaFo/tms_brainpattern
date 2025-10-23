import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Simulation parameters
fs = 250  # Hz
duration = 30  # seconds
n_samples = fs * duration
time = np.arange(n_samples) / fs

# HMM parameters
n_states = 8
stickiness = 0.95  # probability to stay in same state

# Get first 8 colors from seaborn tab20
palette = sns.color_palette("tab20")[:8]

# Simulate sticky HMM state sequence
states = np.zeros(n_samples, dtype=int)
states[0] = np.random.randint(0, n_states)

for t in range(1, n_samples):
    if np.random.rand() < stickiness:
        states[t] = states[t-1]
    else:
        states[t] = np.random.randint(0, n_states)

# Simulate EEG-like signal in [-1,1]
# Each state is a sinusoid with random phase and amplitude in [-1,1]
signal = np.zeros(n_samples)
freqs = np.linspace(5, 15, n_states)  # different frequency per state

for s in range(n_states):
    idx = states == s
    phase = np.random.rand() * 2*np.pi
    amp = np.random.uniform(0.3, 0.9)  # amplitude close to 1
    signal[idx] = amp * np.sin(2 * np.pi * freqs[s] * time[idx] + phase)

# Ensure signal stays in [-1,1] (just in case)
signal = np.clip(signal, -1, 1)

# Plot
plt.figure(figsize=(15, 4))
plt.plot(time, signal, color='black', label='Simulated EEG', alpha=0.5)

# Overlay states with color shading
for s in range(n_states):
    plt.fill_between(time, -1.1, 1.1, where=(states==s), color=palette[s], alpha=0.5)

plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.ylim([-1.2, 1.2])
plt.savefig('eeg_with_hmm_overlay.svg')
plt.show()
