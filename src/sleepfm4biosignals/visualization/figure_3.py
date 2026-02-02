import re
import numpy as np
import pandas as pd
from io import StringIO
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

from statistics import NormalDist

from figure_3_data import dataset_configs, data_ECG

# Ensure a consistent font style
plt.rcParams["font.family"] = "serif"

def create_combination_metric_df(

    raw_text: str,

    metric: str = "balanced_accuracy",

    conf_level: float = 0.95,

    recalculate_balanced_accurracy: bool = False,

) -> pd.DataFrame:

    """Parse the multi-combination log and return per-channel summaries with confidence intervals."""

    lines = [line.strip() for line in raw_text.strip().splitlines() if line.strip()]

    pattern = re.compile(

        r"(?P<channels>\d+)\s+channels,\s+combination\s+(?P<combination>\d+).*?:\s*(?P<metrics>.+)"

    )

    rows = []

    for line in lines:

        match = pattern.match(line)

        if not match:

            continue

        entry = {

            "channels": int(match.group("channels")),

            "combination": int(match.group("combination")),

        }

        metrics_blob = match.group("metrics")

        for metric_pair in metrics_blob.split(","):

            if ":" not in metric_pair:

                continue

            key, value = metric_pair.split(":", 1)

            try:

                entry[key.strip()] = float(value.strip())

            except ValueError:

                continue

        rows.append(entry)

    if not rows:

        raise ValueError("No valid rows found in combination log.")

    combo_df = pd.DataFrame(rows).sort_values(["channels", "combination"]).reset_index(drop=True)

    if recalculate_balanced_accurracy:
        print("Recalculating balanced accuracy from sensitivity and specificity.")

        required_cols = {"sensitivity", "specificity"}

        missing = [col for col in required_cols if col not in combo_df.columns]

        if missing:

            raise ValueError(

                f"Cannot recalculate balanced accuracy because columns {missing} are missing from the data."

            )

        combo_df["balanced_accuracy"] = (

            combo_df["sensitivity"] + combo_df["specificity"]

        ) / 2.0

    if metric not in combo_df.columns:

        raise ValueError(f"Metric '{metric}' not found in parsed combination data.")

    summary = (

        combo_df.groupby("channels")[metric]

        .agg(["mean", "std", "count"])

        .rename(columns={"mean": metric})

        .reset_index()

    )

    summary["std"] = summary["std"].fillna(0.0)

    summary["stderr"] = summary.apply(

        lambda row: row["std"] / (row["count"] ** 0.5) if row["count"] > 0 else 0.0,

        axis=1,

    )

    alpha = max(0.0, min(1.0, 1 - conf_level))

    z_score = NormalDist().inv_cdf(1 - alpha / 2) if conf_level < 1 else 0.0

    summary["ci_low"] = (summary[metric] - z_score * summary["stderr"]).clip(lower=0.0)

    summary["ci_high"] = (summary[metric] + z_score * summary["stderr"]).clip(upper=1.0)

    summary["metric"] = metric

    summary["count"] = summary["count"].astype(int)

    return summary

def plot_combination_ci(summary_df: pd.DataFrame, dataset_name: str = "", metric: str = "balanced_accuracy"):

    """Plot the per-channel mean with its confidence interval."""

    if metric not in summary_df.columns:

        raise ValueError(f"Column '{metric}' not found in dataframe.")

    required_cols = {"channels", "ci_low", "ci_high"}

    if not required_cols.issubset(summary_df.columns):

        raise ValueError(f"Dataframe must include columns: {required_cols}")

    ordered = summary_df.sort_values("channels")

    with plt.style.context("default"):

        fig, ax = plt.subplots(figsize=(7, 4.5))

        ax.plot(

            ordered["channels"],

            ordered[metric],

            color="#1f77b4",

            marker="o",

            linewidth=2,

            label="Mean performance",

        )

        ax.fill_between(

            ordered["channels"],

            ordered["ci_low"],

            ordered["ci_high"],

            color="#1f77b4",

            alpha=0.2,

            label="95% CI",

        )

        ax.set_xlabel("Number of Channels")

        ax.set_ylabel(metric.replace("_", " ").title())

        title_dataset = f"{dataset_name} " if dataset_name else ""

        ax.set_title(f"{title_dataset}{metric.replace('_', ' ').title()} vs. Channels (Combinations)")

        ax.set_xticks(ordered["channels"].unique())

        ax.grid(True, axis="y", linestyle="--", alpha=0.4)

        ax.legend()

        if dataset_name:

            fig.savefig(

                f"{dataset_name}_{metric}_combination_ci.pdf",

                bbox_inches="tight",

            )

        return ax


def add_icon_to_title(ax, icon_path):
    # Load the image
    img = plt.imread(icon_path)
    # Zoom adjusts the size of your icon
    imagebox = OffsetImage(img, zoom=0.05) 
    
    # Position the icon relative to the axes (1.0 is the top, 0.5 is the center)
    ab = AnnotationBbox(imagebox, (0.95, 1.08), xycoords='axes fraction',
                        frameon=False, box_alignment=(0.4, 0.4))
    ax.add_artist(ab)


def plot_brain_heart_grid(
    brain_configs,
    ecg_text,
    *,
    ecg_order=None,
    brain_icon_path="icons/brain-icon.png",
    heart_icon_path="icons/heartbeat-icon.png",
    highlight_ratio=0.95,
    ncols=4,
    sharey=True,
):
    """
    Creates a grid of plots showing performance scaling by channel count.
    
    Highlights markers in red if they achieve >= highlight_ratio of the baseline/peak performance.
    """
    
    ecg_df = pd.read_csv(StringIO(ecg_text), sep="\t")
    ecg_df.columns = [col.strip() for col in ecg_df.columns]
    ecg_df = ecg_df.sort_values("channels")

    # 2. Process Brain (EEG) Entries
    brain_entries = []
    for config in brain_configs:
        # Note: create_combination_metric_df is assumed to be defined in your environment
        summary = create_combination_metric_df(
            config["raw_text"],
            metric=config.get("metric", "Bal. Acc."),
            recalculate_balanced_accurracy=config.get("recalculate_balanced_accurracy", False),
        )

        ordered = summary.sort_values("channels")
        brain_entries.append({
            "kind": "brain",
            "name": config.get("dataset_name", "EEG"),
            "metric_key": config.get("metric", "Bal. Acc."),
            "data": ordered,
        })

        

    # 3. Process Heart (ECG) Entries
    ecg_order = ecg_order or [col for col in ecg_df.columns if col != "channels"]
    heart_entries = []
    for dataset in ecg_order:
        heart_entries.append({
            "kind": "heart",
            "name": dataset.replace(" F1", ""),
            "metric_key": "F1-Score",
            "series": ecg_df[dataset],
            "channels": ecg_df["channels"].to_numpy(),
        })

    entries = brain_entries + heart_entries
    nrows = (len(entries) + ncols - 1) // ncols
    
    # Using the 'Modern Academic' Palette
    primary_color = "#457B9D"   # Steel Blue
    highlight_color = "#E63946" # Crimson
    base_marker_color = "#A8DADC" # Soft Teal

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, sharey=False, 
                             figsize=(3.5 * ncols, 3 * nrows))
    axes_list = list(np.atleast_1d(axes).flat)

    # 4. Plotting Loop
    for idx, (ax, entry) in enumerate(zip(axes_list, entries)):
        if entry["kind"] == "brain":
            x, y = entry["data"]["channels"], entry["data"][entry["metric_key"]]
            ci_low, ci_high = entry["data"]["ci_low"], entry["data"]["ci_high"]
            
            # Highlight logic: relative to full channel capacity (max x)
            baseline = y.iloc[np.argmax(x)]
            ax.fill_between(x, ci_low, ci_high, color=primary_color, alpha=0.15)
        else:
            x, y = entry["channels"], entry["series"]
            baseline = y.max()

        # Shared Plotting Logic
        threshold = highlight_ratio * baseline
        colors = [highlight_color if val >= threshold else base_marker_color for val in y]
        
        ax.plot(x, y, color=primary_color, linewidth=2, zorder=2)
        ax.scatter(x, y, c=colors, edgecolor=primary_color, linewidth=0.5, s=40, zorder=3)

        # 5. Styling and Annotation

        metric_to_display = entry["metric_key"].replace("_", " ").title()
        if metric_to_display.lower() in ["bal. acc.", "balanced accuracy"]:
            metric_to_display = "Bal. Acc."
        elif metric_to_display.lower() =='auroc':
            metric_to_display = "AUROC"
        ax.set_title(f"{entry['name']} ({metric_to_display})", 
                     fontsize=13)
        ax.set_xlabel("Number of Channels", fontsize=12)
        ax.grid(True, axis="both", linestyle="--", alpha=0.5)

        ax.set_xticks(x)
        ax.set_xticklabels(x, fontsize=8)
        
        if not sharey or (idx % ncols == 0):
            ax.set_ylabel("Score", fontsize=9)

        if idx != 2 and idx != 3:
            ax.set_ylim(0.59, 1.0)
        else:
            ax.set_ylim(0.35, 1.0)

        # Add Icon via AnnotationBbox
        if entry["name"] in ['CPSC', 'G12EC', 'PTB-XL']:
            add_icon_to_title(ax, '/home/wls_braincapture_dk/SleepPT4Biosignals/src/sleeppt4biosignals/heart-beat.png')
        else:    
            add_icon_to_title(ax, '/home/wls_braincapture_dk/SleepPT4Biosignals/src/sleeppt4biosignals/brain.png')

    # Clean up empty subplots
    for ax in axes_list[len(entries):]:
        ax.axis("off")

    plt.tight_layout()
    plt.savefig("figures/Brain_Heart_scaling.pdf", bbox_inches="tight", dpi=300)
    plt.close()


if __name__ == "__main__":
    plot_brain_heart_grid(dataset_configs, data_ECG)