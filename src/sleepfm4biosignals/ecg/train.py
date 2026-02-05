#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
train_sleepfm.py

SleepFM finetuning on CODE-15 (public layout):
- exams.csv with columns: exam_id, trace_file, and label columns (--labels)
- HDF5 parts: exams_partXX.hdf5 in a folder (or passed as files)
- Each HDF5 has datasets: exam_id (N,), tracings (N, 4096, 12)

Outputs:
- out/args.json
- out/history.csv
- out/model_best.pth
"""

import os
import re
import glob
import json
import random
import argparse
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score, confusion_matrix, precision_recall_curve

from from sleepfm4biosignals.eeg.model import SetTransformer


# -------------------------
# Basics
# -------------------------
def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def parse_part_id(path_or_name: str) -> Optional[int]:
    m = re.search(r"exams_part(\d+)\.hdf5$", os.path.basename(str(path_or_name)))
    return int(m.group(1)) if m else None


def expand_hdf5_inputs(items: List[str]) -> List[str]:
    out: List[str] = []
    for it in items:
        it = os.path.expanduser(it)
        if os.path.isdir(it):
            out.extend(glob.glob(os.path.join(it, "exams_part*.hdf5")))
        else:
            out.append(it)
    out = [p for p in out if os.path.isfile(p)]
    return list(dict.fromkeys(sorted(out)))


def load_sleepfm_encoder_weights(encoder: nn.Module, ckpt_path: str, device: torch.device) -> None:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            state = ckpt["state_dict"]
        elif "model" in ckpt:
            state = ckpt["model"]
        else:
            state = ckpt
    else:
        state = ckpt

    cleaned = {}
    for k, v in state.items():
        k = k[len("module."):] if k.startswith("module.") else k
        if k.startswith("encoder."):
            cleaned[k.split(".", 1)[1]] = v
        elif k.startswith("backbone."):
            cleaned[k.split(".", 1)[1]] = v
        else:
            cleaned[k] = v

    missing, unexpected = encoder.load_state_dict(cleaned, strict=False)
    print(f"[pretrain] missing={len(missing)} unexpected={len(unexpected)}")


# -------------------------
# HDF5 index
# -------------------------
@dataclass(frozen=True)
class H5Loc:
    row: int


def build_exam_index_by_part(hdf5_by_part: Dict[int, str], ignore_exam_id: int = 0) -> Dict[int, Dict[int, H5Loc]]:
    idx_by_part: Dict[int, Dict[int, H5Loc]] = {}
    for part, path in sorted(hdf5_by_part.items()):
        with h5py.File(path, "r") as f:
            if "exam_id" not in f or "tracings" not in f:
                raise KeyError(f"{path} must contain 'exam_id' and 'tracings'. Found: {list(f.keys())}")

            exam_ids = f["exam_id"][:].astype(np.int64, copy=False)
            part_map: Dict[int, H5Loc] = {}
            for i, eid in enumerate(exam_ids):
                eid = int(eid)
                if eid == ignore_exam_id:
                    continue
                if eid not in part_map:
                    part_map[eid] = H5Loc(row=i)

        idx_by_part[part] = part_map
        print(f"[index] part={part:02d} indexed={len(part_map):,} file={os.path.basename(path)}")
    return idx_by_part


# -------------------------
# Dataset
# -------------------------
ExamKey = Tuple[int, int]  # (part_id, exam_id)


class ECGDataset(Dataset):
    def __init__(
        self,
        exam_keys: List[ExamKey],
        labels: np.ndarray,
        idx_by_part: Dict[int, Dict[int, H5Loc]],
        hdf5_by_part: Dict[int, str],
        traces_dset: str,
        orig_freq: float,
        target_freq: float,
        patch_size: int,
        num_patches: int,
        trim_zeros: bool,
    ):
        super().__init__()
        self.exam_keys = exam_keys
        self.labels = labels.astype(np.float32, copy=False)
        self.idx_by_part = idx_by_part
        self.hdf5_by_part = hdf5_by_part
        self.traces_dset = traces_dset

        self.orig_freq = float(orig_freq)
        self.target_freq = float(target_freq)
        self.patch_size = int(patch_size)
        self.num_patches = int(num_patches)
        self.target_len = self.patch_size * self.num_patches
        self.trim_zeros = bool(trim_zeros)

        self._handles: Dict[int, h5py.File] = {}

        if len(self.exam_keys) != len(self.labels):
            raise ValueError(f"exam_keys ({len(self.exam_keys)}) != labels ({len(self.labels)})")

    def __len__(self) -> int:
        return len(self.exam_keys)

    def __getstate__(self):
        d = dict(self.__dict__)
        d["_handles"] = {}
        return d

    def _handle(self, part: int) -> h5py.File:
        if part not in self._handles:
            self._handles[part] = h5py.File(self.hdf5_by_part[part], "r")
        return self._handles[part]

    def _read_trace(self, key: ExamKey) -> np.ndarray:
        part, exam_id = int(key[0]), int(key[1])
        loc = self.idx_by_part[part].get(exam_id, None)
        if loc is None:
            raise KeyError(f"(part={part}, exam_id={exam_id}) not found")

        f = self._handle(part)
        x = np.asarray(f[self.traces_dset][loc.row], dtype=np.float32)  # (4096,12) typically
        if x.ndim != 2:
            raise ValueError(f"Expected 2D tracings, got {x.shape}")

        if x.shape[-1] == 12:
            x = x.T  # (12,T)
        elif x.shape[0] == 12:
            pass
        else:
            raise ValueError(f"Expected 12 leads, got {x.shape}")

        return x

    def _resample(self, x_ct: torch.Tensor) -> torch.Tensor:
        if self.orig_freq == self.target_freq:
            return x_ct
        C, T = x_ct.shape
        T_new = int(round(T * self.target_freq / self.orig_freq))
        T_new = max(1, T_new)
        x = x_ct.unsqueeze(0)
        x = F.interpolate(x, size=T_new, mode="linear", align_corners=False)
        return x.squeeze(0)

    def _center_crop_or_pad(self, x_ct: torch.Tensor):
        C, T = x_ct.shape
        L = self.target_len
        if T == L:
            return x_ct, torch.zeros((C, L), dtype=torch.long)
        if T > L:
            start = (T - L) // 2
            x_out = x_ct[:, start:start + L]
            return x_out, torch.zeros((C, L), dtype=torch.long)

        pad_total = L - T
        pad_left = pad_total // 2
        pad_right = pad_total - pad_left
        x_out = F.pad(x_ct, (pad_left, pad_right), mode="constant", value=0.0)

        mask = torch.zeros((C, L), dtype=torch.long)
        if pad_left > 0:
            mask[:, :pad_left] = 1
        if pad_right > 0:
            mask[:, -pad_right:] = 1
        return x_out, mask

    def __getitem__(self, idx: int):
        key = self.exam_keys[idx]
        y = self.labels[idx]

        x = torch.from_numpy(self._read_trace(key)).float()

        if self.trim_zeros:
            eps = 1e-7
            nz = (x.abs() > eps).any(dim=0)
            if nz.any():
                first = int(nz.float().argmax().item())
                last = int((nz.shape[0] - 1) - torch.flip(nz, dims=[0]).float().argmax().item())
                if last > first + 10:
                    x = x[:, first:last + 1]

        x = self._resample(x)
        x, mask = self._center_crop_or_pad(x)

        return x, mask, torch.from_numpy(y)


# -------------------------
# Model
# -------------------------
def _padding(kernel_size: int) -> int:
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd")
    return kernel_size // 2


class ResBlockToken1D(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int = 7, dropout: float = 0.1):
        super().__init__()
        p = _padding(kernel_size)
        self.conv1 = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=p, bias=False)
        self.bn1 = nn.BatchNorm1d(embed_dim)
        self.conv2 = nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=p, bias=False)
        self.bn2 = nn.BatchNorm1d(embed_dim)
        self.act = nn.ReLU()
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        y = x
        x = self.drop(self.act(self.bn1(self.conv1(x))))
        x = self.bn2(self.conv2(x))
        return self.drop(self.act(x + y))


class TokenResBlocksHead(nn.Module):
    def __init__(self, embed_dim: int, n_classes: int, n_blocks: int = 2, kernel_size: int = 7, dropout: float = 0.2):
        super().__init__()
        p = _padding(kernel_size)
        self.stem = nn.Sequential(
            nn.Conv1d(embed_dim, embed_dim, kernel_size, padding=p, bias=False),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.blocks = nn.Sequential(*[
            ResBlockToken1D(embed_dim, kernel_size=kernel_size, dropout=dropout) for _ in range(n_blocks)
        ])
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, z):
        z = self.blocks(self.stem(z))
        return self.classifier(z.mean(dim=-1))


class SleepFMECGModel(nn.Module):
    def __init__(
        self,
        n_channels: int,
        n_classes: int,
        patch_size: int,
        max_seq_length: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        pooling_head: int,
        dropout: float,
        head_type: str,
        head_blocks: int,
        head_kernel_size: int,
        head_dropout: float,
    ):
        super().__init__()
        self.encoder = SetTransformerGeneral(
            in_channels=n_channels,
            patch_size=patch_size,
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            pooling_head=pooling_head,
            dropout=dropout,
            max_seq_length=max_seq_length,
        )
        self.head_type = head_type
        if head_type == "linear":
            self.head = nn.Linear(embed_dim, n_classes)
        elif head_type == "resblocks":
            self.head = TokenResBlocksHead(
                embed_dim=embed_dim,
                n_classes=n_classes,
                n_blocks=head_blocks,
                kernel_size=head_kernel_size,
                dropout=head_dropout,
            )
        else:
            raise ValueError(f"Unknown head_type: {head_type}")

    def forward(self, x, mask=None):
        pooled, embedding = self.encoder(x, mask)
        if self.head_type == "linear":
            return self.head(pooled.mean(dim=1))
        if embedding.ndim != 4:
            raise ValueError(f"Expected embedding ndim=4, got {tuple(embedding.shape)}")
        z = embedding.mean(dim=1).transpose(1, 2)  # (B,E,S)
        return self.head(z)


# -------------------------
# Metrics
# -------------------------
@torch.no_grad()
def collect_logits_and_targets(model, loader, device, scale_multiplier: float):
    model.eval()
    logits_all, y_all = [], []
    for x, mask, y in tqdm(loader, desc="eval", leave=False):
        if scale_multiplier != 1.0:
            x = x * float(scale_multiplier)
        x = x.to(device, non_blocking=True)
        logits_all.append(model(x, mask).detach().cpu())
        y_all.append(y.numpy())
    return torch.cat(logits_all, dim=0), np.vstack(y_all)


def compute_metrics_from_logits(logits: torch.Tensor, y_true: np.ndarray, labels: List[str]):
    y_prob = torch.sigmoid(logits).cpu().numpy()
    y_true = y_true.astype(np.float32)

    best_thr: Dict[str, float] = {}
    for i, name in enumerate(labels):
        p, r, t = precision_recall_curve(y_true[:, i], y_prob[:, i])
        f = 2 * (p * r) / (p + r + 1e-8)
        idx = int(np.nanargmax(f[:-1])) if len(f) > 1 else 0
        best_thr[name] = float(t[idx]) if len(t) > 1 else 0.5

    y_pred = np.zeros_like(y_prob, dtype=int)
    for i, name in enumerate(labels):
        y_pred[:, i] = (y_prob[:, i] >= best_thr[name]).astype(int)

    out: Dict[str, float] = {}
    for i, name in enumerate(labels):
        yt, yp = y_true[:, i], y_pred[:, i]
        out[f"{name}_f1"] = f1_score(yt, yp, zero_division=0)
        out[f"{name}_sens"] = recall_score(yt, yp, zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yt, yp, labels=[0, 1]).ravel()
        out[f"{name}_spec"] = tn / (tn + fp) if (tn + fp) > 0 else np.nan
        out[f"{name}_thr"] = best_thr[name]

    out["accuracy"] = accuracy_score(y_true.flatten(), y_pred.flatten())

    aucs = []
    for i in range(y_true.shape[1]):
        if len(np.unique(y_true[:, i])) > 1:
            aucs.append(roc_auc_score(y_true[:, i], y_prob[:, i]))
    out["auc_macro"] = float(np.nanmean(aucs)) if len(aucs) else float("nan")

    out["f1_macro"] = float(np.mean([out[f"{n}_f1"] for n in labels]))
    out["sens_macro"] = float(np.nanmean([out[f"{n}_sens"] for n in labels]))
    out["spec_macro"] = float(np.nanmean([out[f"{n}_spec"] for n in labels]))
    return out


# -------------------------
# Main
# -------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="exams.csv")
    ap.add_argument("--hdf5", nargs="+", required=True, help="HDF5 parts or directory with exams_part*.hdf5")
    ap.add_argument("--out", required=True, help="Output directory")

    ap.add_argument("--labels", nargs="+", default=["1dAVb", "RBBB", "LBBB", "SB", "AF", "ST"])
    ap.add_argument("--traces_dset", default="tracings")
    ap.add_argument("--exam_id_col", default="exam_id")
    ap.add_argument("--trace_file_col", default="trace_file")

    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--seed", type=int, default=2)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--val_frac", type=float, default=0.10)

    ap.add_argument("--orig_freq", type=float, default=400.0)
    ap.add_argument("--target_freq", type=float, default=400.0)
    ap.add_argument("--patch_size", type=int, default=128)
    ap.add_argument("--num_patches", type=int, default=30)
    ap.add_argument("--trim_zeros", action="store_true")
    ap.add_argument("--scale_multiplier", type=float, default=1.0)

    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--num_heads", type=int, default=4)
    ap.add_argument("--num_layers", type=int, default=3)
    ap.add_argument("--pooling_head", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_seq_length", type=int, default=128)

    ap.add_argument("--head_type", type=str, default="resblocks", choices=["linear", "resblocks"])
    ap.add_argument("--head_blocks", type=int, default=2)
    ap.add_argument("--head_kernel_size", type=int, default=7)
    ap.add_argument("--head_dropout", type=float, default=0.2)

    ap.add_argument("--encoder_lr", type=float, default=1e-4)
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--warmup_epochs", type=int, default=3)
    ap.add_argument("--patience", type=int, default=4)
    ap.add_argument("--min_lr", type=float, default=1e-7)
    ap.add_argument("--lr_factor", type=float, default=0.1)

    ap.add_argument("--pretrained_ckpt", type=str, default=None)
    ap.add_argument("--no_pretrain", action="store_true")

    ap.add_argument("--cuda", action="store_true")
    ap.add_argument("--num_workers", type=int, default=-1)

    args = ap.parse_args()
    seed_everything(args.seed)

    device = torch.device("cuda" if (args.cuda and torch.cuda.is_available()) else "cpu")
    if args.num_workers < 0:
        args.num_workers = min(8, multiprocessing.cpu_count())
    pin = (device.type == "cuda")

    out_dir = os.path.abspath(args.out)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2, sort_keys=True)

    hdf5_paths = expand_hdf5_inputs(args.hdf5)
    if len(hdf5_paths) == 0:
        raise ValueError("No HDF5 files found.")

    hdf5_by_part: Dict[int, str] = {}
    for p in hdf5_paths:
        part = parse_part_id(p)
        if part is None:
            raise ValueError(f"Bad HDF5 name: {p} (expected exams_partXX.hdf5)")
        hdf5_by_part[part] = p

    idx_by_part = build_exam_index_by_part(hdf5_by_part, ignore_exam_id=0)

    df = pd.read_csv(args.csv)
    for c in [args.exam_id_col, args.trace_file_col, *args.labels]:
        if c not in df.columns:
            raise KeyError(f"CSV missing column '{c}'")

    df[args.exam_id_col] = pd.to_numeric(df[args.exam_id_col], errors="coerce").fillna(0).astype(np.int64)
    df = df[df[args.exam_id_col] != 0].copy()

    df["part_id"] = df[args.trace_file_col].astype(str).apply(parse_part_id)
    df = df[df["part_id"].isin(hdf5_by_part.keys())].copy()
    df["part_id"] = df["part_id"].astype(int)

    df["exam_key"] = list(zip(df["part_id"].astype(int), df[args.exam_id_col].astype(int)))

    def exists(k: ExamKey) -> bool:
        p, eid = int(k[0]), int(k[1])
        return (p in idx_by_part) and (eid in idx_by_part[p])

    before = len(df)
    df = df[df["exam_key"].apply(exists)].copy()
    print(f"[data] rows={len(df):,} (dropped {before - len(df):,})")

    if len(df) < 2:
        raise ValueError("Not enough samples after filtering.")

    y = df[args.labels].astype(np.float32).values
    keys: List[ExamKey] = df["exam_key"].tolist()

    N = len(df)
    idxs = np.arange(N, dtype=np.int64)
    rng = np.random.RandomState(args.seed)
    rng.shuffle(idxs)

    n_val = int(round(N * float(args.val_frac)))
    n_val = max(1, min(n_val, N - 1))
    val_pos = idxs[:n_val]
    tr_pos = idxs[n_val:]

    tr_keys = [keys[i] for i in tr_pos]
    va_keys = [keys[i] for i in val_pos]
    tr_y = y[tr_pos]
    va_y = y[val_pos]

    train_ds = ECGDataset(
        exam_keys=tr_keys,
        labels=tr_y,
        idx_by_part=idx_by_part,
        hdf5_by_part=hdf5_by_part,
        traces_dset=args.traces_dset,
        orig_freq=args.orig_freq,
        target_freq=args.target_freq,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        trim_zeros=args.trim_zeros,
    )
    valid_ds = ECGDataset(
        exam_keys=va_keys,
        labels=va_y,
        idx_by_part=idx_by_part,
        hdf5_by_part=hdf5_by_part,
        traces_dset=args.traces_dset,
        orig_freq=args.orig_freq,
        target_freq=args.target_freq,
        patch_size=args.patch_size,
        num_patches=args.num_patches,
        trim_zeros=args.trim_zeros,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=pin)
    valid_loader = DataLoader(valid_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=pin)

    x0, _, _ = train_ds[0]
    n_channels = int(x0.shape[0])
    target_len = int(x0.shape[1])
    print(f"[shape] C={n_channels} L={target_len} device={device}")

    model = SleepFMECGModel(
        n_channels=n_channels,
        n_classes=len(args.labels),
        patch_size=args.patch_size,
        max_seq_length=args.max_seq_length,
        embed_dim=args.embed_dim,
        num_heads=args.num_heads,
        num_layers=args.num_layers,
        pooling_head=args.pooling_head,
        dropout=args.dropout,
        head_type=args.head_type,
        head_blocks=args.head_blocks,
        head_kernel_size=args.head_kernel_size,
        head_dropout=args.head_dropout,
    ).to(device)

    if not args.no_pretrain:
        if not args.pretrained_ckpt:
            raise ValueError("Provide --pretrained_ckpt or set --no_pretrain.")
        load_sleepfm_encoder_weights(model.encoder, args.pretrained_ckpt, device)

    enc_params = list(model.encoder.parameters())
    head_params = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    optimizer = optim.Adam(
        [{"params": enc_params, "lr": args.encoder_lr}, {"params": head_params, "lr": args.head_lr}]
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=args.patience, factor=args.lr_factor, min_lr=args.min_lr
    )
    loss_fn = nn.BCEWithLogitsLoss()

    best_val = float("inf")
    best_path = os.path.join(out_dir, "model_best.pth")
    hist_path = os.path.join(out_dir, "history.csv")

    target_enc_lr = float(args.encoder_lr)
    target_head_lr = float(args.head_lr)

    rows = []

    for ep in range(args.epochs):
        if args.warmup_epochs and ep < args.warmup_epochs:
            warm = float(ep + 1) / float(args.warmup_epochs)
            optimizer.param_groups[0]["lr"] = target_enc_lr * warm
            optimizer.param_groups[1]["lr"] = target_head_lr * warm
        elif args.warmup_epochs and ep == args.warmup_epochs:
            optimizer.param_groups[0]["lr"] = target_enc_lr
            optimizer.param_groups[1]["lr"] = target_head_lr

        model.train()
        tr_sum, tr_n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Epoch {ep:03d} train", leave=False)
        for x, mask, yb in pbar:
            if args.scale_multiplier != 1.0:
                x = x * float(args.scale_multiplier)

            x = x.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(x, mask)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()

            bs = x.size(0)
            tr_sum += float(loss.item()) * bs
            tr_n += bs
            pbar.set_postfix(loss=tr_sum / max(1, tr_n))
        tr_loss = tr_sum / max(1, tr_n)

        model.eval()
        va_sum, va_n = 0.0, 0
        pbar = tqdm(valid_loader, desc=f"Epoch {ep:03d} valid", leave=False)
        with torch.no_grad():
            for x, mask, yb in pbar:
                if args.scale_multiplier != 1.0:
                    x = x * float(args.scale_multiplier)

                x = x.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                logits = model(x, mask)
                loss = loss_fn(logits, yb)

                bs = x.size(0)
                va_sum += float(loss.item()) * bs
                va_n += bs
                pbar.set_postfix(loss=va_sum / max(1, va_n))
        va_loss = va_sum / max(1, va_n)

        logits_va, y_va = collect_logits_and_targets(model, valid_loader, device, args.scale_multiplier)
        metrics = compute_metrics_from_logits(logits_va, y_va, args.labels)

        lr_enc = optimizer.param_groups[0]["lr"]
        lr_head = optimizer.param_groups[1]["lr"]

        row = {
            "epoch": ep,
            "train_loss": tr_loss,
            "valid_loss": va_loss,
            "lr_encoder": lr_enc,
            "lr_head": lr_head,
            **metrics,
        }
        rows.append(row)
        pd.DataFrame(rows).to_csv(hist_path, index=False)

        if va_loss < best_val:
            best_val = va_loss
            torch.save(
                {"epoch": ep, "model": model.state_dict(), "valid_loss": va_loss, "optimizer": optimizer.state_dict(), "args": vars(args)},
                best_path,
            )

        if not args.warmup_epochs or ep >= args.warmup_epochs:
            scheduler.step(va_loss)

        print(
            f"Epoch {ep:03d} train={tr_loss:.4f} valid={va_loss:.4f} "
            f"lr_enc={lr_enc:.1e} lr_head={lr_head:.1e} "
            f"auc={metrics['auc_macro']:.3f} f1={metrics['f1_macro']:.3f}"
        )

    print(f"[done] best_valid_loss={best_val:.6f} -> {best_path}")


if __name__ == "__main__":
    main()

