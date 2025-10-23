# %%
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.gridspec as gridspec
from pathlib import Path

# File paths
hads_path = Path('C:/Users/Carina/Desktop/QIMR/TMS_MDD/hads_scores.csv')
madrs_path = Path('C:/Users/Carina/Desktop/QIMR/TMS_MDD/madrs_scores.csv')
icons_path = Path('C:/Users/Carina/Desktop/QIMR/TMS_MDD/icons_black')

# Load data
madrs_df = pd.read_csv(madrs_path)
hads_df = pd.read_csv(hads_path)

# Constants
RESPONDER_COLOR = 'mediumorchid'
NON_RESPONDER_COLOR = 'black'
HADS_SESSIONS = ['pre', 'week 1', 'week 2', 'week 3', 'week 4', 'post']
HADS_TICKS = list(range(len(HADS_SESSIONS)))


# %%


# ========== TIMELINE PLOTTING ==========
def plot_timeline(ax, fontsize=12):
    
    timeline_pos = ['Pre-Treatment', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Post-Treatment']
    icons = {
        "EEG": load_icon(os.path.join(icons_path, "activity.png"), zoom=1),
        "TMS": load_icon(os.path.join(icons_path, "zap.png"), zoom=1),
        "HADS": load_icon(os.path.join(icons_path, "hads.png"), zoom=1),
        "MADRS": load_icon(os.path.join(icons_path, "madrs.png"), zoom=1)
    }

    x_pos = [i * 2 for i in range(len(timeline_pos))]
    ax.set_xlim(-1, max(x_pos) + 1)
    ax.set_ylim(0, 6)
    ax.axis("off")

    ax.hlines(y=3, xmin=x_pos[0], xmax=x_pos[-1], color='black', linewidth=2)
    for x, label in zip(x_pos, timeline_pos):
        ax.plot(x, 3, 'o', color='black')
        ax.text(x, 2.6, label, ha='center', fontsize=fontsize)

    for i in range(len(x_pos) - 1):
        mid = (x_pos[i] + x_pos[i+1]) / 2
        ax.add_artist(AnnotationBbox(icons["TMS"], (mid, 3.5), frameon=False))
        if i == 0:
            ax.text(mid + 0.4, 3.5, "TMS", va='center', fontsize=fontsize)

    for i in [0, 2, 4]:
        if i < len(x_pos) - 1:
            mid = (x_pos[i] + x_pos[i + 1]) / 2
            ax.add_artist(AnnotationBbox(icons["EEG"], (mid, 4.0), frameon=False))
            if i == 0:
                ax.text(mid + 0.4, 4.0, "EEG", va='center', fontsize=fontsize)

    for i in [0, 5]:
        x = x_pos[i]
        ax.add_artist(AnnotationBbox(icons["MADRS"], (x - 0.1, 1.5), frameon=False))
        if i == 0:
            ax.text(x + 0.2, 1.5, "MADRS", fontsize=fontsize, va='center')

    for i, x in enumerate(x_pos):
        ax.add_artist(AnnotationBbox(icons["HADS"], (x - 0.1, 2.0), frameon=False))
        if i == 0:
            ax.text(x + 0.2, 2.0, "HADS", fontsize=fontsize, va='center')


# %%
def plot_individual_lines(ax, df, sessions, color_responder, color_nonresponder, alpha=0.4):
    for _, row in df.iterrows():
        color = color_responder if row['responder'] else color_nonresponder
        y_vals = [row[s] for s in sessions]
        ax.plot(HADS_TICKS[:len(sessions)], y_vals, color=color, alpha=alpha)

def plot_group_means(ax, df, sessions, color_responder, color_nonresponder, linestyle='-', linewidth=2):
    grouped = df.groupby("responder")
    for responder, group in grouped:
        color = color_responder if responder else color_nonresponder
        means = group[sessions].mean().values
        sems = group[sessions].sem().values
        ax.errorbar(HADS_TICKS[:len(sessions)], means, yerr=sems, fmt='o',
                    linestyle=linestyle, color=color, linewidth=linewidth,
                    label='Responder' if responder else 'Non-Responder')

# %%
# Plot MADRS only
fig, ax = plt.subplots()
plot_individual_lines(ax, madrs_df, ['pre', 'post'], RESPONDER_COLOR, NON_RESPONDER_COLOR)
plot_group_means(ax, madrs_df, ['pre', 'post'], RESPONDER_COLOR, NON_RESPONDER_COLOR, linestyle='')
ax.set_xlim(-0.5, 1.5)
ax.set_ylim(0, 50)
ax.set_xticks([0, 1])
ax.set_xticklabels(['Pre', 'Post'], fontname='Arial', fontsize=10)
ax.set_ylabel('MADRS Score', fontname='Arial', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)

# %%
# Plot HADS only
fig, ax = plt.subplots()
plot_individual_lines(ax, hads_df, HADS_SESSIONS, RESPONDER_COLOR, NON_RESPONDER_COLOR, alpha=0.25)
plot_group_means(ax, hads_df, HADS_SESSIONS, RESPONDER_COLOR, NON_RESPONDER_COLOR)
ax.set_xlim(-0.5, 5.5)
ax.set_ylim(0, 25)
ax.set_xticks(HADS_TICKS)
ax.set_xticklabels(['Pre', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Post'], fontsize=10, fontname='Arial')
ax.set_ylabel('HADS Score', fontname='Arial', fontsize=12)
ax.spines[['top', 'right']].set_visible(False)

# %%
# Combined MADRS + HADS plot
fig = plt.figure(figsize=(8, 4), layout='constrained')
gs = gridspec.GridSpec(2, 3, figure=fig)

# Timeline (top, spans all 3 columns)
ax_timeline = fig.add_subplot(gs[0, :])
plot_timeline(ax_timeline, fontsize=12)

ax1 = fig.add_subplot(gs[1, 0])
ax2 = fig.add_subplot(gs[1, 1:])

# MADRS (left)
plot_individual_lines(ax1, madrs_df, ['pre', 'post'], RESPONDER_COLOR, NON_RESPONDER_COLOR)
plot_group_means(ax1, madrs_df, ['pre', 'post'], RESPONDER_COLOR, NON_RESPONDER_COLOR, linestyle='')
ax1.set_xlim(-0.5, 1.5)
ax1.set_ylim(0, 50)
ax1.set_xticks([0, 1])
ax1.set_xticklabels(['Pre', 'Post'], fontname='Arial', fontsize=10)
ax1.set_ylabel('MADRS Score', fontname='Arial', fontsize=12)
ax1.spines[['top', 'right']].set_visible(False)
ax1.legend(loc='lower left', fontsize=8, frameon=False)

# HADS (right)
plot_individual_lines(ax2, hads_df, HADS_SESSIONS, RESPONDER_COLOR, NON_RESPONDER_COLOR, alpha=0.25)
plot_group_means(ax2, hads_df, HADS_SESSIONS, RESPONDER_COLOR, NON_RESPONDER_COLOR)
ax2.set_xlim(-0.5, 5.5)
ax2.set_ylim(0, 25)
ax2.set_xticks(HADS_TICKS)
ax2.set_xticklabels(['Pre', 'Week 1', 'Week 2', 'Week 3', 'Week 4', 'Post'], fontsize=10, fontname='Arial')
ax2.set_ylabel('HADS Score', fontname='Arial', fontsize=12)
ax2.spines[['top', 'right']].set_visible(False)

# Optionally save
fig.savefig("madrs_hads_combined.svg", bbox_inches="tight", dpi=300)
plt.show()

# ========== UTILS ==========
def convert_svg_to_png():
    import os
    import cairosvg

    # Path to your SVG folder
    svg_folder = icons_path

    # Loop through files in the folder
    for filename in os.listdir(svg_folder):
        if filename.lower().endswith(".svg"):
            svg_path = os.path.join(svg_folder, filename)
            png_path = os.path.join(svg_folder, filename[:-4] + ".png")
            
            # Convert SVG to PNG
            cairosvg.svg2png(url=svg_path, write_to=png_path)

            print(f"Converted: {filename} → {os.path.basename(png_path)}")

def load_icon(path, zoom=0.5):
    return OffsetImage(plt.imread(path), zoom=zoom)
