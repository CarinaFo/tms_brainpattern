"""Plot group-average HMM networks.

"""
# Run in osld_tf environment
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import pickle
import seaborn as sns

plot_psds = True
plot_pow_maps = True
plot_coh_nets = True
plot_coh_maps = False
plot_pow_vs_coh = True
plot_trans_prob = True
plot_sum_stats = True
plot_state_time_course = False

# set working directory
os.chdir(Path('/home/carinaf/tms_mdd'))

basedir = os.getcwd()
save_dir = Path(f'{basedir}/57patients_newmodels_giles_plots')

# Source reconstruction files
mask_file = "MNI152_T1_8mm_brain.nii.gz"
parcellation_file = "fmri_d100_parcellation_with_PCC_reduced_2mm_ss5mm_ds8mm.nii.gz"

# New order for states
#order = [3, 5, 7, 0, 1, 6, 9, 8, 4, 2] # Chet reordered based on power
n_states = [6]
n_sessions = 6

for state in n_states:
    save_dir_plots = f'{save_dir}/plots_{state}'
    os.makedirs(save_dir_plots, exist_ok=True)
    # Get tab20 as a list of colors
    colors = sns.color_palette("tab20", n_colors=state)  

    for ses in range(n_sessions):

        if plot_psds:
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 20,
                "xtick.labelsize": 18,
                "ytick.labelsize": 18,
                "legend.fontsize": 28,
                "lines.linewidth": 6,
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
                    y_range=[0, 0.4],
                    plot_kwargs={"color": "black", "linestyle": "--"},
                    fig_kwargs={"figsize": (4, 6)} 
                )
                ax.plot(f, p[i], color=colors[i]) #*wb_comp[0,:]
                ax.set_yticks([0, 0.2, 0.4])
                plt.show()
                plotting.save(fig, f"{save_dir_plots}/psd_{i:02d}_{state}_{ses}.svg", 
                tight_layout=True)

        if plot_pow_maps:
            from osl_dynamics.analysis import power
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 16,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
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
                plot_kwargs={'hemispheres': ['left'], 'views': ['lateral']}, #'vmin': -0.02,  'vmax': 0.05, 
                #'cmap': 'coolwarm'}, #'PuOr'
                filename=f"{save_dir_plots}/pow_{ses}.png",
                component=0
            )

        if plot_coh_nets:
            from osl_dynamics.analysis import connectivity
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
            c = connectivity.mean_coherence_from_spectra(f, gcoh, wb_comp)[0]

            # Subtract reference and threshold
            gfo = np.average(fo, axis=0, weights=w)
            c -= np.average(c, axis=0, weights=gfo)
            c = connectivity.threshold(c, percentile=95, absolute_value=True)

            # Plot
            connectivity.save(
                c,
                parcellation_file=parcellation_file,
                plot_kwargs={"display_mode": "z", "annotate": False, 'edge_cmap': 'coolwarm'},  # ← Ensures two decimal places},
                filename=f"{save_dir_plots}/coh_{ses}.png",
                component=0
            )

        if plot_coh_maps:
            from osl_dynamics.analysis import connectivity, power
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 16,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
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
                "axes.labelsize": 16,
                "xtick.labelsize": 16,
                "ytick.labelsize": 16,
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
                filename=f"{save_dir_plots}/pow_vs_coh_parcels_{ses}_{state}.svg",
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
                filename=f"{save_dir_plots}/pow_vs_coh_subjects_{ses}_{state}.svg",
                title='subjects'
            )

        if plot_trans_prob:
            from mpl_toolkits.axes_grid1 import make_axes_locatable

            # Load data
            tp = np.load(f"{save_dir}/tp_{ses}_{state}.npy")[0]
            #tp = tp[np.ix_(order, order)]

            # Extract the diagonal
            diag = np.diag(tp.copy())
            np.fill_diagonal(tp, 0)

            # Plot off diagonals
            fig, ax = plt.subplots()

            im = ax.matshow(tp)

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            fig.colorbar(im, cax=cax, orientation="vertical")
            cax.tick_params(labelsize=17)

            ax.tick_params(labelsize=17)
            ax.set_xlabel("State: To", fontsize=18)
            ax.set_ylabel("State: From", fontsize=18)
            ax.xaxis.set_label_position("top")

            plt.savefig(f"{save_dir_plots}/trans_prob_{ses}_{state}.png")
            plt.close()

            # Plot the diagonal
            fig, ax = plt.subplots(figsize=(4,8))

            im = ax.matshow(diag[:, np.newaxis])

            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="50%", pad=0.25)
            fig.colorbar(im, cax=cax, orientation="vertical")
            cax.tick_params(labelsize=24)

            ax.set_xticklabels([""])
            ax.set_yticklabels([""] + ["1", "3", "5", "7", "9"])
            ax.tick_params(labelsize=24)
            ax.set_ylabel("State", fontsize=24)

            plt.savefig(f"{save_dir_plots}/trans_prob_diag_{ses}_{state}.png")
            plt.close()

        if plot_sum_stats:
            from osl_dynamics.utils import plotting

            plotting.set_style({
                "axes.labelsize": 16,
                "xtick.labelsize": 14,
                "ytick.labelsize": 14,
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
                    fo, lt, intv, sr, cmap=colors, filename=f"{save_dir_plots}/sum_stats__{ses}_{state}.svg",
                )

            # Plot the distribution of fractional occupancy (FO) across subjects
            plotting.plot_violin(fo.T, x_label="State", y_label="FO", sns_kwargs={'palette': colors},
                                filename=f"{save_dir_plots}/fo_{ses}_{state}.svg")


        if plot_state_time_course:
            
            from osl_dynamics.inference import modes
            from osl_dynamics.utils import plotting

            from matplotlib import cm
            from matplotlib.colors import to_hex

            # load stc
            stc = pickle.load(open(f"{save_dir}/states_{ses}_{state}.pkl", 'rb'))
            # Calculate a state time course by taking the most likely state
            stc = modes.argmax_time_courses(stc)

            if state == 12:
                for idx, s in enumerate(stc):
                    plotting.plot_alpha(s, cmap='tab20',
                    filename=f"{save_dir_plots}/stc_{idx}_{ses}_{state}.png")
            else:
                for idx, s in enumerate(stc):
                    plotting.plot_alpha(s,
                    filename=f"{save_dir_plots}/stc_{idx}_{ses}_{state}.png")
