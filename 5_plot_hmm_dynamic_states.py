"""Plot group-average HMM networks.

"""
# Run in osld environment on linux or windows
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns

system='linux'

plot_psds = True
plot_pow_maps = True
plot_coh_nets = True
plot_coh_maps = False
plot_pow_vs_coh = True
plot_trans_prob = True
plot_sum_stats = True
plot_state_time_course = True

if system == 'linux':
    # set working directory
    base_dir = Path('/home/carinaf/canonical_hmm_finalsample')
elif system == 'windows':
    # Windows home dir
    base_dir = Path("L:/Lab_LucaC/Carina/")
else:
    "No available system path defined *windows* or *linux*"

save_dir = Path(f'{base_dir}/hmm_fits_05Hzcanonical_1Hzfiltered')

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# New order for states
#order = [3, 5, 7, 0, 1, 6, 9, 8, 4, 2] # Chet reordered based on power
n_states = [10]
n_sessions = 1

for state in n_states:
    save_dir_plots = f'{save_dir}/plots_{state}'
    os.makedirs(save_dir_plots, exist_ok=True)
    # Get tab20 as a list of colors
    colors = sns.color_palette("tab20", n_colors=state)  

    for ses in range(n_sessions):

        if plot_psds:
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 18,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "legend.fontsize": 22,
                "lines.linewidth": 10,
            })

            # Load data
            f = np.load(Path(f"{save_dir}/f_{ses}_{state}.npy"))
            psd = np.load(Path(f"{save_dir}/psd_{ses}_{state}.npy"))#[:, order]
            w = np.load(Path(f"{save_dir}/w_{ses}_{state}.npy"))
            fo = np.load(f"{save_dir}/fo_{ses}_{state}.npy") #[:, order]
            wb_comp = np.load(Path(f"{save_dir}/nnmf_{ses}_{state}.npy"))

            # Group average PSD for each state
            gpsd = np.average(psd, axis=0, weights=w)
            gfo = np.average(fo, axis=0, weights=w)
            mgpsd = np.average(gpsd, axis=0, weights=gfo)
            p = np.mean(gpsd, axis=1)
            mp = np.mean(mgpsd, axis=0)

            # Plot
            for i in range(p.shape[0]):
                fig, ax = plotting.plot_line(
                    [f],
                    [mp],   # *wb_comp[0,:],
                    x_label="Frequency (Hz)",
                    y_label="PSD (a.u.)",
                    x_range=[f[0], f[-1]],
                    y_range=[0, 0.35],
                    plot_kwargs={"color": "black", "linestyle": "--"},
                    #fig_kwargs={"figsize": (8, 12)} 
                )
                ax.plot(f, p[i], color=colors[i]) #*wb_comp[0,:]
                ax.set_yticks([0, 0.1, 0.2, 0.3])
                ax.set_xticks([10, 20, 30, 40])
                plt.show()
                plotting.save(fig, f"{save_dir_plots}/psd_{i:02d}_{ses}.png", 
                tight_layout=True)

        if plot_pow_maps:
            from osl_dynamics.analysis import power
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 18,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
            })

            # Load data
            f = np.load(Path(f"{save_dir}/f_{ses}_{state}.npy"))
            psd = np.load(Path(f"{save_dir}/psd_{ses}_{state}.npy"))#[:, order]
            w = np.load(Path(f"{save_dir}/w_{ses}_{state}.npy"))
            fo = np.load(f"{save_dir}/fo_{ses}_{state}.npy") #[:, order]

            # Calculate group average
            gpsd = np.average(psd, axis=0, weights=w)
            p = power.variance_from_spectra(f, gpsd, components=wb_comp)
            gfo = np.average(fo, axis=0, weights=w)

            # Plot
            power.save(
                p,
                mask_file=mask_file,
                parcellation_file=parcellation_file,
                subtract_mean=True,
                mean_weights=gfo,
                #plot_kwargs={'hemispheres': ['left'], 'views': ['lateral'], 'cbar_tick_format': '%.2f'},
                #'cmap': 'coolwarm'}, #'PuOr'
                filename=f"{save_dir_plots}/pow_{ses}_.png",
                component=0
            )

        if plot_coh_nets:
            from osl_dynamics.analysis import connectivity
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 18,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
            })

            # Load data
            f = np.load(Path(f"{save_dir}/f_{ses}_{state}.npy"))
            coh = np.load(Path(f"{save_dir}/coh_{ses}_{state}.npy"))#[:, order]
            w = np.load(Path(f"{save_dir}/w_{ses}_{state}.npy"))
            fo = np.load(f"{save_dir}/fo_{ses}_{state}.npy") #[:, order]

            # Calculate group average
            gcoh = np.average(coh, axis=0, weights=w)
            c = connectivity.mean_coherence_from_spectra(f, gcoh, wb_comp)[0]

            # Subtract reference and threshold
            gfo = np.average(fo, axis=0, weights=w)
            c -= np.average(c, axis=0, weights=gfo)
            c = connectivity.threshold(c, percentile=95, absolute_value=True)
            # round for plotting
            c_rounded = c.astype(float).round(3)

            # Plot
            connectivity.save(
                c_rounded,
                parcellation_file=parcellation_file,
                #plot_kwargs={"display_mode": "z", "annotate": False, 'edge_cmap': 'coolwarm', 'colorbar': False},  
                filename=f"{save_dir_plots}/coh_{ses}_.png"
            )

        if plot_coh_maps:
            from osl_dynamics.analysis import connectivity, power
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
            })

            # Load data
            f = np.load(Path(f"{save_dir}/f_{ses}_{state}.npy"))
            coh = np.load(Path(f"{save_dir}/coh_{ses}_{state}.npy"))#[:, order]
            w = np.load(Path(f"{save_dir}/w_{ses}_{state}.npy"))
            fo = np.load(f"{save_dir}/fo_{ses}_{state}.npy") #[:, order]


            # Calculate group average
            gcoh = np.average(coh, axis=0, weights=w)
            c = connectivity.mean_coherence_from_spectra(f, gcoh)
            c = connectivity.mean_connections(c)
            gfo = np.average(fo, axis=0, weights=w)

            # Plot
            power.save(
                c,
                mask_file=mask_file,
                parcellation_file=parcellation_file,
                subtract_mean=True,
                mean_weights=gfo,
                filename="mean_coh_.png",
            )

        if plot_pow_vs_coh:
            from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
            from osl_dynamics.analysis import power, connectivity
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 14,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
                "legend.fontsize": 12,
            })

            # Load data
            f = np.load(Path(f"{save_dir}/f_{ses}_{state}.npy"))
            coh = np.load(Path(f"{save_dir}/coh_{ses}_{state}.npy"))#[:, order]
            w = np.load(Path(f"{save_dir}/w_{ses}_{state}.npy"))
            fo = np.load(f"{save_dir}/fo_{ses}_{state}.npy") #[:, order]


            # Calculate power/coherence
            p = power.variance_from_spectra(f, psd)
            c = connectivity.mean_coherence_from_spectra(f, coh)
            c = connectivity.mean_connections(c)

            # Calculate group average
            p_ = np.mean(p, axis=0)
            c_ = np.mean(c, axis=0)

            # Plot
            plotting.plot_scatter(
                p_,
                c_,
                labels=[f"State {i}" for i in range(1, state+1)],
                x_label="Power",
                y_label="Coherence",
                filename=f"{save_dir_plots}/pow_vs_coh_parcels_{ses}_{state}.png",
                title='parcel'
            )

            # Calculate average over parcels
            p_ = np.mean(p, axis=-1).T
            c_ = np.mean(c, axis=-1).T

            # Plot
            plotting.plot_scatter(
                p_,
                c_,
                labels=[f"State {i}" for i in range(1, state+1)],
                x_label="Power (a.u.)",
                y_label="Mean Coherence",
                filename=f"{save_dir_plots}/pow_vs_coh_subjects_{ses}_{state}.png",
                title='subjects'
            )

        if plot_trans_prob:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            # Load data
            tp = np.load(f"{save_dir}/tp_{ses}_{state}.npy")
            tp_mean = tp.mean(axis=0)

            # Extract the diagonal
            diag = np.diag(tp_mean.copy())

            # Mask the diagonal for off-diagonal plot
            tp_masked = tp_mean.copy()
            np.fill_diagonal(tp_masked, np.nan)

            # Define state labels (1 to 8)
            states = np.arange(1, state+1)

            # ---------- Plot off-diagonal ----------
            fig, ax = plt.subplots(figsize=(7, 6))

            # Plot off-diagonal transitions
            im = ax.matshow(tp_masked, cmap="viridis", vmin=np.nanmin(tp_masked), vmax=np.nanmax(tp_masked))

            # Overlay grey squares for diagonal cells
            for i in range(len(states)):
                ax.add_patch(plt.Rectangle((i - 0.5, i - 0.5), 1, 1, color="lightgrey", zorder=2))

            # Colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            cax.tick_params(labelsize=18)

            # Axis labels and ticks
            ax.tick_params(labelsize=18)
            ax.set_xticks(np.arange(len(states)))
            ax.set_yticks(np.arange(len(states)))
            ax.set_xticklabels(states)
            ax.set_yticklabels(states)
            ax.xaxis.set_label_position("top")
            ax.set_xlabel("Next State", fontsize=22)
            ax.set_ylabel("Current State", fontsize=22)
            ax.set_title("Transition Probability Matrix", fontsize=22, pad=20)

            plt.tight_layout()
            plt.savefig(f"{save_dir_plots}/trans_prob_{ses}_{state}.png")
            plt.show()
            plt.close()


            # ---------- Plot diagonal (self transitions) ----------
            fig, ax = plt.subplots(figsize=(4, 8))

            # Plot diagonal as a single-column heatmap
            im = ax.matshow(diag[:, np.newaxis], cmap="viridis")

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="50%", pad=0.25)
            fig.colorbar(im, cax=cax, orientation="vertical")
            cax.tick_params(labelsize=18)

            ax.set_xticks([0])
            ax.set_xticklabels(["Self"])
            ax.set_yticks(np.arange(len(states)))
            ax.set_yticklabels(states)
            ax.tick_params(labelsize=18)
            ax.set_ylabel("State", fontsize=22)
            ax.set_title("Self-Transitions", fontsize=22, pad=20)

            plt.tight_layout()
            plt.savefig(f"{save_dir_plots}/trans_prob_diag_{ses}_{state}.png")
            plt.show()
            plt.close()

        if plot_sum_stats:
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 18,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
            })

                # Load
            fo = np.load(f"{save_dir}/fo_{ses}_{state}.npy")
            lt = np.load(f"{save_dir}/lt_{ses}_{state}.npy")
            intv = np.load(f"{save_dir}/intv_{ses}_{state}.npy")
            sr = np.load(f"{save_dir}/sr_{ses}_{state}.npy")

            if state == 12:
                from matplotlib import cm
                from matplotlib.colors import to_hex
                # Get tab20 colormap
                tab20 = cm.get_cmap("tab20")
                # Pick first 12 colors
                palette = [to_hex(tab20(i)) for i in range(state)]

                # Plot
                plotting.plot_hmm_summary_stats(
                    fo, lt, intv, sr, cmap=palette, filename=f"{save_dir_plots}/sum_stats__{ses}_{state}.svg",
                )
            else:
                plotting.plot_hmm_summary_stats(
                    fo, lt, intv, sr, cmap=colors, filename=f"{save_dir_plots}/sum_stats__{ses}_{state}.png",
                )

            # Plot the distribution of fractional occupancy (FO) across subjects
            plotting.plot_violin(fo.T, x_label="HMM State", y_label="Fractional Occupancies", sns_kwargs={'palette': colors},
                                filename=f"{save_dir_plots}/fo_{ses}_{state}.png")


        if plot_state_time_course:
            
            from osl_dynamics.inference import modes
            from osl_dynamics.utils import plotting

            from matplotlib import cm
            from matplotlib.colors import to_hex

            # Version incompatibility fix
            import sys
            # Temporary alias for compatibility with old pickles
            sys.modules['numpy._core'] = np
            sys.modules['numpy._core.multiarray'] = np.core.multiarray
            sys.modules['numpy._core.numeric'] = np.core.numeric

            import pickle

            # load stc
            stc = pickle.load(open(f"{save_dir}/states_{ses}_{state}.pkl", 'rb'))
            # Calculate a state time course by taking the most likely state
            stc = modes.argmax_time_courses(stc)
            import random
            indices = random.sample(range(0, 40), 10)

            if state == 12:
                for idx in indices:
                    plotting.plot_alpha(stc[idx], cmap='tab20',
                    filename=f"{save_dir_plots}/stc_{idx}_{ses}_{state}.svg")
            else:
                for idx in indices:
                    plotting.plot_alpha(stc[idx], cmap='tab20',
                    filename=f"{save_dir_plots}/stc_{idx}_{ses}_{state}.png")
