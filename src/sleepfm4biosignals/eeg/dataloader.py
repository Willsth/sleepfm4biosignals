# from eegmodel.dataset import H5PYDatasetLabeled

# Reach out to {researcher}, {email address} for collaboration and licensing.
# Name and email address are placeholders and should be replaced with actual contact information after blinded review.

import torch
import json
from torch.utils.data import DataLoader
from sklearn.model_selection import GroupShuffleSplit
from typing import Optional, Generator

def get_dataloaders(
    train_path: str,
    test_path: Optional[str] = None,
    transformer = None,
    batch_size: int = 32,
    num_workers: int = 4,
    seed: int = 42,
    n_splits: int = 5,
    train_balance_strategy: Optional[str] = None,
    split_info_path: str = "splits.json"
) -> Generator:
    """Generates fold-based dataloaders and saves subject split metadata."""
    base_ds = H5PYDatasetLabeled(train_path, transform=transformer)
    splitter = GroupShuffleSplit(n_splits=n_splits, test_size=0.1, random_state=seed)
    
    split_metadata = {}

    for fold, (train_val_idx, fold_test_idx) in enumerate(splitter.split(base_ds, groups=base_ds.subjects)):
        inner_split = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        t_idx, v_idx = next(inner_split.split(train_val_idx, groups=base_ds.subjects[train_val_idx]))
        
        # Convert to global indices
        actual_train_idx = torch.from_numpy(train_val_idx[t_idx])
        actual_val_idx = torch.from_numpy(train_val_idx[v_idx])
        actual_test_idx = torch.from_numpy(fold_test_idx)

        # Log metadata
        split_metadata[f"fold_{fold+1}"] = {
            "train_subjects": sorted(list(set(base_ds.subjects[actual_train_idx].tolist()))),
            "val_subjects": sorted(list(set(base_ds.subjects[actual_val_idx].tolist()))),
            "test_subjects": "external" if test_path else sorted(list(set(base_ds.subjects[actual_test_idx].tolist())))
        }

        # Create Loaders
        train_ds = H5PYDatasetLabeled(train_path, transform=transformer, balance_dataset=train_balance_strategy)
        train_ds.update_index_map(actual_train_idx)
        train_ds.apply_balancing()

        val_ds = H5PYDatasetLabeled(train_path, transform=transformer)
        val_ds.update_index_map(actual_val_idx)

        if test_path:
            test_ds = H5PYDatasetLabeled(test_path, transform=transformer)
        else:
            test_ds = H5PYDatasetLabeled(train_path, transform=transformer)
            test_ds.update_index_map(actual_test_idx)

        yield (
            fold,
            DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
            DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers),
            DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        )

    with open(split_info_path, "w") as f:
        json.dump(split_metadata, f, indent=4)