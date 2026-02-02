import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

# Data provided in your request
EEG_sleepfm = {
    "TUEV": {"metric": "Balanced Accuracy", "no_pt": {"mean": 0.5585, "std": 0.0168}, "pt": {"mean": 0.6612, "std": 0.0252}, "pt+pe": {"mean": 0.6387, "std": 0.0267}, "sota": {"mean": 0.6759, "std": 0.0229}},
    "TUAB": {"metric": "Balanced Accuracy", "no_pt": {"mean": 0.7886, "std": 0.0439}, "pt": {"mean": 0.8255, "std": 0.0023}, "pt+pe": {"mean": 0.8303, "std": 0.0074}, "sota": {"mean": 0.8315, "std": 0.0014}},
    "TUSL": {"metric": "AUROC", "no_pt": {"mean": 0.6709, "std": 0.0436}, "pt": {"mean": 0.7179, "std": 0.0388}, "pt+pe": {"mean": 0.7457, "std": 0.0370}, "sota": {"mean": 0.731, "std": 0.012}},
    "BrainCapture": {"metric": "F1-Score", "no_pt": {"mean": 0.6877, "std": 0.0437}, "pt": {"mean": 0.8019, "std": 0.0604}, "pt+pe": {"mean": 0.8666, "std": 0.0220}, "sota": {"mean": 0.7274, "std": 0.0216}},
    "CHB-MIT": {"metric": "Balanced Accuracy", "no_pt": {"mean": 0.6678, "std": 0.0581}, "pt": {"mean": 0.7180, "std": 0.0512}, "pt+pe": {"mean": 0.7468, "std": 0.0333}, "sota": {"mean": 0.7398, "std": 0.0284}},
}

EKG_sleepfm = {
    "PTB-XL": {"metric": "F1-Score", "no_pt": {"mean": 0.782, "std": 0.066}, "pt": {"mean": 0.807, "std": 0.071}, "sota": {"mean": 0.803, "std": 0.107}},
    "G12EC": {"metric": "F1-Score", "no_pt": {"mean": 0.691, "std": 0.045}, "pt": {"mean": 0.713, "std": 0.023}, "sota": {"mean": 0.763, "std": 0.028}},
    "CPSC": {"metric": "F1-Score", "no_pt": {"mean": 0.759, "std": 0.071}, "pt": {"mean": 0.770, "std": 0.068}, "sota": {"mean": 0.784, "std": 0.086}},
}

def plot_combined_eeg_ekg(eeg_data, ekg_data, save_path="Figure 2.pdf"):
    # Define colors and labels in the desired order
    # Note: SOTA is last so it appears on the far right
    config = [
        ("no_pt", "w/o PT", "#A8DADC"),
        ("pt", "PT", "#457B9D"),
        ("pt+pe", "PT+PE", "#1D3557"),
        ("sota", "SOTA", "#E63946")
    ]
    
    # Merge data into a list of dictionaries for easier iteration
    all_data = []
    for d, cat in [(eeg_data, "EEG"), (ekg_data, "ECG")]:
        for name, vals in d.items():
            all_data.append({"name": name, "cat": cat, "vals": vals})

    fig, ax = plt.subplots(figsize=(28, 6))
    width = 0.18
    x_indices = np.arange(len(all_data))

    # To keep track of legend handles uniquely
    legend_map = {}

    for i, entry in enumerate(all_data):
        # Identify which categories from 'config' actually exist for this dataset
        available = [c for c in config if c[0] in entry["vals"]]
        
        # Calculate starting offset to center the group of bars over the X-tick
        # total_width = len(available) * width
        start_offset = -((len(available) - 1) * width) / 2

        for j, (key, label, color) in enumerate(available):
            data_point = entry["vals"][key]
            mean = data_point["mean"]
            std = data_point.get("std", 0)
            
            pos = i + start_offset + (j * width)
            bar = ax.bar(pos, mean, width, yerr=std, color=color, capsize=3, edgecolor='white', linewidth=0.5)
            
            # Store bar for legend (only once per label)
            if label not in legend_map:
                legend_map[label] = bar

            # Add text labels on top
            ax.text(pos, mean + (std if std else 0) + 0.01, f"{mean:.2f}", 
                    ha='center', va='bottom', fontsize=16, fontweight='bold')

    # Formatting
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"{e['name']}\n({e['vals']['metric']})" for e in all_data], rotation=0, fontsize=16)
    ax.set_title("Fine-Tuning Performance on EEG and ECG Downstream Tasks", fontsize=32, y=1.05)
    ax.set_ylabel("Metric", fontsize=18)
    ax.tick_params(axis='y', labelsize=16)
    ax.set_ylim(0.4, 1.0)
    ax.grid(axis='y', linestyle='--', alpha=0.5)

    # Category Dividers (EEG vs ECG)
    h = ax.get_ylim()[1] * 0.93
    eeg_count = len(eeg_data)
    ax.axvline(eeg_count - 0.5, color="black", linewidth=1, linestyle="-")
    ax.text(eeg_count/2 - 0.5, h, "EEG", ha='center', fontsize=26)
    ax.text(eeg_count + (len(ekg_data)/2) - 0.5, h, "ECG", ha='center', fontsize=26)
    # Legend
    ax.legend(legend_map.values(), legend_map.keys(), loc='upper left', bbox_to_anchor=(0, 1), frameon=False, ncol=4, fontsize=16)

    heart_path = 'heart-beat.png'
    brain_path = 'brain.png'

    heart_img = plt.imread(heart_path)
    heart_imagebox = OffsetImage(heart_img, zoom=0.07) 

    brain_img = plt.imread(brain_path)
    brain_imagebox = OffsetImage(brain_img, zoom=0.07)
    
    # Position the icon relative to the axes (1.0 is the top, 0.5 is the center)
    heart_ab = AnnotationBbox(heart_imagebox, (0.833, 0.92), xycoords='axes fraction',
                        frameon=False, box_alignment=(0.5, 0.5))

    brain_ab = AnnotationBbox(brain_imagebox, (0.355, 0.92), xycoords='axes fraction',
                        frameon=False, box_alignment=(0.5, 0.5))
    
    ax.add_artist(heart_ab)
    ax.add_artist(brain_ab)

    plt.tight_layout()

    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()


if __name__ == "__main__":
    plt.rcParams["font.family"] = "serif"
    plot_combined_eeg_ekg(EEG_sleepfm, EKG_sleepfm)