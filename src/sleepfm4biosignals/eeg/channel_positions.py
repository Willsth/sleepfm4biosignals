import torch
import numpy as np
from typing import List, Union

# Standard 10-20 EEG system positions (x, y, z coordinates)
# x: left(-) to right(+), y: posterior(-) to anterior(+), z: inferior(-) to superior(+)

STANDARD_1020_POSITIONS = {
    # Frontal Pole
    "Fp1": [-0.31, 0.95, 0.00],
    "Fp2": [0.31, 0.95, 0.00],
    "Fpz": [0.00, 0.95, 0.00],
    
    # Frontal
    "F7": [-0.71, 0.59, -0.39],
    "F3": [-0.42, 0.68, 0.60],
    "Fz": [0.00, 0.72, 0.69],
    "F4": [0.42, 0.68, 0.60],
    "F8": [0.71, 0.59, -0.39],
    
    # Frontal-Central
    "FC5": [-0.71, 0.31, 0.42],
    "FC1": [-0.22, 0.37, 0.90],
    "FC2": [0.22, 0.37, 0.90],
    "FC6": [0.71, 0.31, 0.42],
    
    # Central/Temporal
    "T7": [-0.95, 0.00, -0.31],
    "T3": [-0.95, 0.00, -0.31],  # Alternative name for T7
    "C3": [-0.64, 0.00, 0.77],
    "Cz": [0.00, 0.00, 1.00],
    "C4": [0.64, 0.00, 0.77],
    "T8": [0.95, 0.00, -0.31],
    "T4": [0.95, 0.00, -0.31],  # Alternative name for T8
    
    # Central-Parietal
    "CP5": [-0.71, -0.31, 0.42],
    "CP1": [-0.22, -0.37, 0.90],
    "CP2": [0.22, -0.37, 0.90],
    "CP6": [0.71, -0.31, 0.42],
    
    # Parietal/Temporal
    "T5": [-0.71, -0.59, -0.39],
    "P7": [-0.71, -0.59, -0.39],  # Alternative name for T5
    "P3": [-0.42, -0.68, 0.60],
    "Pz": [0.00, -0.72, 0.69],
    "P4": [0.42, -0.68, 0.60],
    "T6": [0.71, -0.59, -0.39],
    "P8": [0.71, -0.59, -0.39],  # Alternative name for T6
    
    # Occipital
    "O1": [-0.31, -0.95, 0.00],
    "O2": [0.31, -0.95, 0.00],
    "Oz": [0.00, -0.95, 0.00],
    
    # Additional standard positions
    "A1": [-1.00, 0.00, -0.31],  # Left ear/mastoid
    "A2": [1.00, 0.00, -0.31],   # Right ear/mastoid
    "M1": [-1.00, 0.00, -0.31],  # Alternative name for A1
    "M2": [1.00, 0.00, -0.31],   # Alternative name for A2
}

def _get_single_electrode_position(electrode_name: str) -> list:
    """Get 3D position of a single electrode."""
    electrode_clean = electrode_name.strip()  # Don't use .upper()
    
    # Try exact match
    if electrode_clean in STANDARD_1020_POSITIONS:
        return STANDARD_1020_POSITIONS[electrode_clean]
    
    # Try uppercase version
    electrode_upper = electrode_clean.upper()
    if electrode_upper in STANDARD_1020_POSITIONS:
        return STANDARD_1020_POSITIONS[electrode_upper]
    
    # Try title case (first letter capital)
    electrode_title = electrode_clean.capitalize()
    if electrode_title in STANDARD_1020_POSITIONS:
        return STANDARD_1020_POSITIONS[electrode_title]
    
    raise ValueError(
        f"Electrode '{electrode_name}' not found. "
        f"Available: {sorted(STANDARD_1020_POSITIONS.keys())}"
    )

def _parse_channel_name(channel_name: str) -> list:
    """Parse channel name and return 3D position (handles monopolar and bipolar)."""
    if '-' in channel_name:
        # Bipolar channel
        parts = channel_name.split('-')
        electrode_parts = [p.strip() for p in parts if not p.isdigit()]
        
        if len(electrode_parts) >= 2:
            pos1 = np.array(_get_single_electrode_position(electrode_parts[0]))
            pos2 = np.array(_get_single_electrode_position(electrode_parts[1]))
            midpoint = (pos1 + pos2) / 2.0
            return midpoint.tolist()
        elif len(electrode_parts) == 1:
            return _get_single_electrode_position(electrode_parts[0])
        else:
            raise ValueError(f"Could not parse: {channel_name}")
    else:
        # Monopolar channel
        return _get_single_electrode_position(channel_name)


def get_channel_positions(
    channel_names: Union[List[str], tuple, torch.Tensor],
    device: torch.device = None
) -> torch.Tensor:
    """
    Convert channel names to 3D positions (case-insensitive).
    
    Handles monopolar ('FP1') and bipolar ('FP1-F7') channels.
    """
    if isinstance(channel_names, torch.Tensor):
        raise ValueError("channel_names should be list/tuple of strings")
    
    # Handle batched format
    if isinstance(channel_names, (tuple, list)):
        if len(channel_names) > 0 and isinstance(channel_names[0], (tuple, list)):
            channel_names = [ch[0] if isinstance(ch, (tuple, list)) else ch 
                           for ch in channel_names]
    
    positions = []
    for ch_name in channel_names:
        if not isinstance(ch_name, str):
            raise ValueError(f"Expected string, got {type(ch_name)}: {ch_name}")
        
        try:
            pos = _parse_channel_name(ch_name)
            positions.append(pos)
        except ValueError as e:
            raise ValueError(f"Error processing '{ch_name}': {str(e)}")
    
    positions_tensor = torch.tensor(positions, dtype=torch.float32)
    
    if device is not None:
        positions_tensor = positions_tensor.to(device)
    
    return positions_tensor