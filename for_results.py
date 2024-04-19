import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator


def text_color(background_color):
    """
    A simple function to return 'white' or 'black' depending on the perceived
    luminance of the background color.
    """
    try:
        r, g, b, _ = background_color  # Try to unpack RGBA values
    except ValueError:
        r, g, b = background_color  # Unpack RGB values if no alpha channel

    # Calculate the luminance of the color using the formula for luminance
    luminance = 0.299 * r + 0.587 * g + 0.114 * b
    return "white" if luminance < 0.5 else "black"


# Load the results in the "results" directory
results = pd.read_csv("FOR_study_runs.csv")
ram_towards = []
ram_away = []
summaries = []
for i, row in results.iterrows():
    summary_dir = Path(row["result_dir"], "summary.p")
    with open(summary_dir, "rb") as f:
        summary = pickle.load(f)
    summaries.append(summary)
    ko = summary["Sun_ko"]
    toward = 90 - ko[0]
    away = ko[1] - 90
    ram_towards.append(toward)
    ram_away.append(away)

# Assuming the lists `ram_towards`, `ram_away`, and `summaries` have been
# populated accordingly
data = pd.DataFrame(
    {
        "ram_towards": ram_towards,
        "ram_away": ram_away,
        "unique_planets_detected": [
            summary["unique_planets_detected"] for summary in summaries
        ],
        "one_detection": [summary["one_detection"] for summary in summaries],
        "two_detections": [summary["two_detections"] for summary in summaries],
        "three_plus_detections": [
            summary["three_plus_detections"] for summary in summaries
        ],
        "available_planets": [summary["available_planets"] for summary in summaries],
        "planets_in_universe": [
            summary["planets_in_universe"] for summary in summaries
        ],
    }
)
data["schedule_success"] = data["three_plus_detections"] / data["available_planets"]
data["percent_planets_detected"] = (
    data["unique_planets_detected"] / data["planets_in_universe"]
)

# Creating a list of pivot tables and their titles
pivot_tables = [
    (
        "Percent of Planets Observed 3+ Times out of Possible Planets",
        data.pivot_table(
            values="schedule_success",
            index="ram_away",
            columns="ram_towards",
            aggfunc="mean",
            fill_value=0,
        ),
    ),
    (
        "Percent of Planets Detected Once Out of All Planets",
        data.pivot_table(
            values="percent_planets_detected",
            index="ram_away",
            columns="ram_towards",
            aggfunc="mean",
            fill_value=0,
        ),
    ),
    # (
    #     "Available Planets",
    #     data.pivot_table(
    #         values="available_planets",
    #         index="ram_away",
    #         columns="ram_towards",
    #         aggfunc="sum",
    #         fill_value=0,
    #     ),
    # ),
    # (
    #     "One Detection",
    #     data.pivot_table(
    #         values="one_detection",
    #         index="ram_away",
    #         columns="ram_towards",
    #         aggfunc="sum",
    #         fill_value=0,
    #     ),
    # ),
    # (
    #     "Two Detections",
    #     data.pivot_table(
    #         values="two_detections",
    #         index="ram_away",
    #         columns="ram_towards",
    #         aggfunc="sum",
    #         fill_value=0,
    #     ),
    # ),
]

fig, axs = plt.subplots(1, len(pivot_tables), figsize=(6 * len(pivot_tables), 6))
axs = axs.flatten()

max_ticks = 10
cmap = plt.get_cmap("viridis")

# Loop through each pivot table and its corresponding axis
for ax, (title, pivot_table) in zip(axs, pivot_tables):
    if ax.get_subplotspec().is_first_col():
        ax.set_ylabel("Pitch away from Sun")
    if ax.get_subplotspec().is_last_row():
        ax.set_xlabel("Pitch towards Sun")

    # max_value = pivot_table.to_numpy().max()
    # min_value = pivot_table.to_numpy().min()
    # range_value = max_value - min_value
    # if range_value > max_ticks:
    #     tick_step = int(range_value / max_ticks)
    # else:
    #     tick_step = 1
    # ticks = np.arange(min_value, max_value + 1, tick_step)
    #
    # boundaries = np.arange(min_value, max_value + 2) - 0.5
    # norm = BoundaryNorm(boundaries, ncolors=256)

    norm = plt.Normalize(vmin=0, vmax=1)
    cax = ax.matshow(pivot_table, interpolation="nearest", cmap="viridis", norm=norm)
    cbar = fig.colorbar(cax, ax=ax, label="Percent")
    locator = MaxNLocator(nbins=max_ticks)
    cbar.locator = locator
    cbar.update_ticks()
    # cbar.set_ticks(ticks)
    # cbar.set_ticklabels(ticks)

    ax.set_title(title)
    # Set axis ticks and labels
    ax.set_xticks(range(len(pivot_table.columns)))
    ax.set_xticklabels(pivot_table.columns)
    ax.set_yticks(range(len(pivot_table.index)))
    ax.set_yticklabels(pivot_table.index)
    for i in range(pivot_table.shape[0]):
        for j in range(pivot_table.shape[1]):
            c = pivot_table.iloc[i, j]
            cell_color = cmap(norm(c))
            color = text_color(cell_color)
            ax.text(j, i, f"{c:.2f}", va="center", ha="center", color=color)

plt.tight_layout()
plt.savefig("FOR_heatmap.png")
plt.show()
