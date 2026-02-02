import numpy as np
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox

def add_icon_to_title(ax, icon_path):
    # Load the image
    img = plt.imread(icon_path)
    # Zoom adjusts the size of your icon
    imagebox = OffsetImage(img, zoom=0.08) 
    
    # Position the icon relative to the axes (1.0 is the top, 0.5 is the center)
    ab = AnnotationBbox(imagebox, (0.95, 1.08), xycoords='axes fraction',
                        frameon=False, box_alignment=(0.5, 0.5))
    ax.add_artist(ab)

def visualize_dataset_class_distribution():
    dataset_class_distribtion_dict = {
        'TUAB (Train)': {'Normal': 30142, 'Abnormal': 30771},
        'TUAB (Eval)': {'Normal': 3253, 'Abnormal': 2646},
        'TUEV (Train)': {"bckg": 52579, "pled": 6011, "artf": 10265, "eyem": 1012, "gped": 11113, "spsw": 640},
        'TUEV (Eval)': {"bckg": 18940, "pled": 1886, "artf": 1998, "eyem": 329, "gped": 3546, "spsw": 567},
        'TUSL': {'Normal': 1399, 'Complex \n Background': 92, 'Seizure': 92, 'Slowing': 94},
        'CHB-MIT': {'Non-Seizure': 47565, 'Seizure': 220},
        'BrainCapture': {'Normal': 4950, 'Abnormal': 3046},
        'CPSC': {'1dAVb': 828, 'RBBB': 1971, 'LBBB': 274, 'SB': 0, 'AF': 1374},
        'G12EC': {'1dAVb': 769, 'RBBB': 556, 'LBBB': 231, 'SB': 0, 'AF': 570},
        'PTB-XL': {'1dAVb': 0, 'RBBB': 542, 'LBBB': 536, 'SB': 637, 'AF': 1330},
    }

    max_abnormal = max(max(dataset_class_distribtion_dict['TUAB (Train)'].values()),
    max(dataset_class_distribtion_dict['TUAB (Eval)'].values()))

    max_events = max(max(dataset_class_distribtion_dict['TUEV (Train)'].values()),
    max(dataset_class_distribtion_dict['TUEV (Eval)'].values()))

    fig, axes = plt.subplots(2, 5, figsize=(28, 12)) # Increased height for rotated labels
    axes_flat = axes.flatten()
    
    # Global constant for bar width
    BAR_WIDTH = 0.6 
    # Max categories across all datasets (TUEV has 6)
    MAX_CATS = 6 

    dataset_items = list(dataset_class_distribtion_dict.items())

    for idx in range(len(axes_flat)):
        current_ax = axes_flat[idx]
        
        if idx < len(dataset_items):
            dataset_name, class_counts = dataset_items[idx]
            labels = list(class_counts.keys())
            sizes = list(class_counts.values())
            
            # Create numeric x-positions evenly spaced based on MAX_CATS
            x_pos = np.arange(len(labels)) + (MAX_CATS - len(labels)) / 2

            if len(labels) == 2:
                x_pos = [1.8, 3.2]  # Center the two bars in the middle positions

            # Set colors
            if len(class_counts) <= 2:
                colors = ['#A8DADC', '#457B9D'] # Using our professional palette
            else:
                colors = plt.cm.Paired.colors[:len(class_counts)]

            # 1. DRAW BARS with fixed width
            current_ax.bar(x_pos, sizes, width=BAR_WIDTH, color=colors, edgecolor='black', zorder=3)
            
            # 2. FORCE UNIFORM X-AXIS
            # This ensures that even if there are only 2 bars, the axis "room" is the same
            current_ax.set_xlim(-0.5, MAX_CATS - 0.5) 
            
            # 3. FORMAT TICKS
            current_ax.set_xticks(x_pos)
            current_ax.set_xticklabels(labels, fontsize=14, rotation=30, ha='right')
            current_ax.tick_params(axis='y', labelsize=14)

            if idx % 5 == 0:
                current_ax.set_ylabel('Number of Samples', fontsize=20)

            current_ax.set_title(dataset_name, fontsize=32)
            current_ax.grid(axis='y', alpha=0.5, linestyle='--', zorder=0)

            # Value labels
            for i, size in enumerate(sizes):
                current_ax.text(x_pos[i], size + (max(sizes) * 0.01), f"{size:,}", 
                                ha='center', va='bottom', fontsize=16, fontweight='bold')
            
            # Icon Logic
            base = '/home/wls_braincapture_dk/SleepPT4Biosignals/src/sleeppt4biosignals/'
            icon_path = base + 'heart-beat.png' if dataset_name in ['CPSC', 'G12EC', 'PTB-XL'] else base + 'brain.png'
            add_icon_to_title(current_ax, icon_path) # Call your helper here
            
        else:
            current_ax.axis('off')

        if idx in [0, 1]: # TUAB Pair
            current_ax.set_ylim(0, max_abnormal * 1.1)
        elif idx in [2, 3]: # TUEV Pair
            current_ax.set_ylim(0, max_events * 1.1)
        else:
            current_ax.set_ylim(0, max(sizes) * 1.1)

    plt.suptitle('Number of Samples Across Downstream Datasets', fontsize=42, y=1.01)
    plt.tight_layout()
    plt.savefig('Figure 1.pdf', bbox_inches="tight")
    plt.show()


if __name__ == "__main__":

    plt.rcParams["font.family"] = "serif"
    
    visualize_dataset_class_distribution()