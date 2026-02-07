import torch
from typing import Optional, Literal
from pytorch_lightning import LightningModule
from torch import nn
import torch.nn.functional as F
from torchmetrics import Accuracy, AUROC, Recall, Specificity, F1Score, MetricCollection, AveragePrecision

import numpy as np
from torch import Tensor
from typing import Tuple, Optional, Literal, List
import logging

from models import SetTransformer, SetTransformerWithChannelPE

def remove_module_from_state_dict(state_dict):
    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith('module.'):
            new_key = key[len('module.'):]
        else:
            new_key = key
        new_state_dict[new_key] = value
    return new_state_dict

def _make_span_from_seeds(seeds: np.ndarray, span: int, total: None | int = None) -> np.ndarray:
    inds = list()
    for seed in seeds:
        for i in range(seed, seed + span):
            if total is not None and i >= total:
                break
            elif i not in inds:
                inds.append(int(i))
    return np.array(inds)

def _make_mask(shape: Tuple[int, int], prob: float, total: int, span: int, allow_no_inds: bool = False) -> torch.Tensor:
    mask = torch.zeros(shape, requires_grad=False, dtype=torch.bool)

    for i in range(shape[0]):
        mask_seeds = list()
        while not allow_no_inds and len(mask_seeds) == 0 and prob > 0:
            mask_seeds = np.nonzero(np.random.rand(total) < prob)[0]

        mask[i, _make_span_from_seeds(mask_seeds, span, total=total)] = True

    return mask

class Permute(nn.Module):
    def __init__(self, *axes: int) -> None:
        super().__init__()
        self.axes = axes

    def forward(self, x: Tensor) -> Tensor:
        return x.permute(*self.axes)
    
class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

class EncodingAugment(nn.Module):
    def __init__(
        self, encoder_height=128, mask_p_t=0.1, mask_p_c=0.01, mask_t_span=6, mask_c_span=64, dropout=0.1, position_encoder=25
    ):
        super().__init__()
        self.mask_replacement = torch.nn.Parameter(torch.zeros(encoder_height), requires_grad=True)
        self.p_t = mask_p_t
        self.p_c = mask_p_c
        self.mask_t_span = mask_t_span
        self.mask_c_span = mask_c_span
        self.transformer_dim = 3 * encoder_height

        conv = nn.Conv1d(encoder_height, encoder_height, position_encoder, padding=position_encoder // 2, groups=8)
        nn.init.normal_(conv.weight, mean=0, std=2 / self.transformer_dim)
        nn.init.constant_(conv.bias, 0)
        conv = nn.utils.parametrizations.weight_norm(conv, dim=2)
        self.relative_position = nn.Sequential(conv, nn.GELU())

        self.input_conditioning = nn.Sequential(
            Permute([0, 2, 1]),
            nn.LayerNorm(encoder_height),
            nn.Dropout(dropout),
            Permute([0, 2, 1]),
            nn.Conv1d(encoder_height, self.transformer_dim, 1),
        )

    def forward(self, x, mask_t=None, mask_c=None):

        x = x.clone()

        bs, feat, seq = x.shape

        if self.training:
            if mask_t is None and self.p_t > 0 and self.mask_t_span > 0:
                mask_t = _make_mask((bs, seq), self.p_t, x.shape[-1], self.mask_t_span)
            if mask_c is None and self.p_c > 0 and self.mask_c_span > 0:
                mask_c = _make_mask((bs, feat), self.p_c, x.shape[1], self.mask_c_span)
    
        if mask_t is not None:
            x.transpose(2, 1).contiguous()[mask_t] = self.mask_replacement
        if mask_c is not None:
            x[mask_c] = 0

        x = self.input_conditioning(x + self.relative_position(x))
        return x

class LinearHeadPE(LightningModule):
    def __init__(
        self,
        load_encoder: bool = True,
        encoder_path: Optional[str] = "pretrained_model.pt",
        fine_tune_strategy: Literal["full", "frozen"] = "full",
        out_features: int = 2,
        use_channel_pe: bool = False,
        # hyperparameters
        encoder_height: int = 128,
        pool_length: int = 4,
        lr: float = 1e-4,
        dropout_rate: float = 0.4,
        label_smoothing: float = 0.1,
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["encoder_path"])
        self.out_features = out_features

        logger = logging.getLogger(__name__)
        logging.basicConfig(level=logging.INFO)
        
        # Initialize Encoder
        if use_channel_pe:
            encoder = SetTransformerWithChannelPE(
                in_channels=1, 
                patch_size=128, 
                embed_dim=128, 
                num_heads=8, 
                num_layers=3, 
                pooling_head=8, 
                dropout=0.3,
                use_channel_pe=True,
                freqs=8
            )
        else:
            encoder = SetTransformer(
                in_channels=1, 
                patch_size=128, 
                embed_dim=128, 
                num_heads=8, 
                num_layers=3, 
                pooling_head=8, 
                dropout=0.3
            )

        # Load checkpoint if needed
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if load_encoder and encoder_path is not None:
            MODEL_PATH = encoder_path
            logger.info(f"Loading encoder from {MODEL_PATH}")

            if device.type == 'cuda':
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cuda'))
                if use_channel_pe:
                    allowed_missing_keys = [
                        "channel_pe_mlp.0.weight", 
                        "channel_pe_mlp.2.weight", 
                        "channel_pe_mlp.2.bias"
                    ]
                    missing_keys, unexpected_keys = encoder.load_state_dict(
                        remove_module_from_state_dict(checkpoint["state_dict"]), 
                        strict=False
                    )
                    for key in missing_keys:
                        if key not in allowed_missing_keys:
                            raise ValueError(f"Unexpected missing key: {key}")
                    if unexpected_keys:
                        raise ValueError(f"Unexpected keys: {unexpected_keys}")
                else: 
                    encoder.load_state_dict(remove_module_from_state_dict(checkpoint["state_dict"]))
            else:
                checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
                encoder.load_state_dict(remove_module_from_state_dict(checkpoint["state_dict"]))

        else:
            logger.info("Random initialization.")

        self.encoder = encoder
            
        if fine_tune_strategy == "frozen":
            for param in self.encoder.parameters():
                param.requires_grad = False
            logger.info("Encoder Frozen.")
        else:
            logger.info("Full fine-tuning enabled.")

        # Heads & Pooling
        self.enc_augment = EncodingAugment(encoder_height=self.hparams.encoder_height)
        self.summarizer = nn.AdaptiveAvgPool1d(self.hparams.pool_length)
        
        head_dim = self.enc_augment.transformer_dim * self.hparams.pool_length

        self.extended_classifier = nn.Sequential(
            Flatten(),
            nn.Linear(head_dim, head_dim),
            nn.Dropout(self.hparams.dropout_rate),
            nn.ReLU(),
            nn.BatchNorm1d(head_dim),
        )

        self.clf = nn.Linear(head_dim, self.out_features)
        
        # Metrics & Loss
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=self.hparams.label_smoothing)
        task = "binary" if self.out_features <= 2 else "multiclass"
        avg = 'macro' if task == "multiclass" else 'micro'

        metrics = MetricCollection({
            "accuracy": Accuracy(task=task, num_classes=self.out_features),
            "balanced_accuracy": Accuracy(task=task, num_classes=self.out_features, average='macro'),
            "sensitivity": Recall(task=task, num_classes=self.out_features, average=avg),
            "specificity": Specificity(task=task, num_classes=self.out_features, average=avg),
            "f1": F1Score(task=task, num_classes=self.out_features, average=avg),
            "weighted_f1": F1Score(task=task, num_classes=self.out_features, average='weighted'),
            "auroc": AUROC(task=task, num_classes=self.out_features, average=avg),
            "auprc": AveragePrecision(task=task, num_classes=self.out_features, average=avg)
        })

        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")
        self.test_metrics = metrics.clone(prefix="test_")

    def forward(self, x, channel_names: Optional[List[str]] = None):
        """
        Args:
            x: (B, C, T) input tensor
            channel_names: List of channel names for positional encoding
        """
        # Pass channel_names to encoder
        _, x = self.encoder(x, channel_names=channel_names)
        x = x.transpose(1, 2)  # (B, C, T)

        x = self.enc_augment(x)
        x = self.summarizer(x)
        x = self.extended_classifier(x)
        return self.clf(x)

    def _shared_step(self, batch, stage: str):
        if len(batch) == 3:
            x, y, channel_names = batch
        else:
            x, y = batch
            channel_names = None
        
        # Forward pass with channel names
        logits = self(x, channel_names=channel_names)
        loss = self.loss_fn(logits, y.long())
        
        # Softmax for probabilities
        probs = torch.softmax(logits, dim=1)
        
        if self.out_features <= 2:
            metric_input = probs[:, 1]
        else:
            metric_input = probs
        
        # Select tracker and update metrics
        metric_tracker = getattr(self, f"{stage}_metrics")
        metric_tracker.update(metric_input, y)
        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def on_train_epoch_end(self):
        self.log_dict(self.train_metrics.compute())
        self.train_metrics.reset()

    def on_validation_epoch_end(self):
        self.log_dict(self.val_metrics.compute())
        self.val_metrics.reset()

    def on_test_epoch_end(self):
        self.log_dict(self.test_metrics.compute())
        self.test_metrics.reset()

    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        return torch.optim.AdamW(trainable_params, lr=self.hparams.lr)