import os
import logging
from typing import Optional

import torch
import wandb
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from dataloader import get_dataloaders
from linearhead import LinearHeadPE

class StandardizeLabel:
	def __init__(self, mean: float = 0.0, std: float = 1e-05, clip: float = 5e-05):
		self.mean = mean
		self.std = std
		self.clip = clip

	def __call__(self, batch: list, training: bool = True) -> torch.Tensor:
		x, y = batch
		x = (torch.clamp(x, -self.clip, self.clip) - self.mean) / self.std
		return [x, y]

def finetune_sleepfm(
	experiment_id: int = 1,
	encoder_path: Optional[str] = None,
	experiment_name: str = "experiment_name",
	project_name: str = "project_name",
	train_path: Optional[str] = None,
	test_path: Optional[str] = None,
	batch_size: int = 16,
	num_workers: int = 8,
	num_epochs: int = 100,
	seed: int = 42,
	device: str = "cuda",
	patience: int = 10,
	n_splits: int = 1,
	out_features: int = 2,
	train_balance_strategy: Optional[str] = None,
	use_channel_pe: bool = False,
    pool_length: int = 2,
	) -> list[dict]:
		"""Fine-tune with balanced training and unbalanced validation/testing."""
		logger = logging.getLogger(__name__)
		logging.basicConfig(level=logging.INFO)

		strategy_map = {1: "full", 2: "full", 4: "frozen"}
		load_enc = experiment_id != 1

		logger.info(f"Using fine-tuning strategy: {strategy_map.get(experiment_id, 'full')}")

		seed_everything(seed, workers=True)

		if wandb.run is not None:
			logger.info("A wandb run is already in progress. Finishing it to start a new one.")
			wandb.finish()

		wandb_logger = WandbLogger(name=experiment_name, project=project_name)
		logger.info(f"Finetuning with parameters: {locals()}")

		# Initialize our generator
		loader_generator = get_dataloaders(
			train_path=train_path,
			test_path=test_path,
			transformer=StandardizeLabel(),
			batch_size=batch_size,
			num_workers=num_workers,
			seed=seed,
			n_splits=n_splits,
			train_balance_strategy=train_balance_strategy,
			split_info_path=f"{experiment_name}_splits.json"
		)

		all_fold_results = []

		for fold, train_loader, val_loader, test_loader in loader_generator:
			fold_name = f"{experiment_name}_fold_{fold + 1}"
			if wandb.run is not None:
				wandb.finish()
			wandb_logger = WandbLogger(name=fold_name, project=project_name)
			logger.info(f"WandB logger initialized for fold: {fold_name}")

			model = LinearHeadPE(
				load_encoder=load_enc,
				encoder_path=encoder_path,
				fine_tune_strategy=strategy_map.get(experiment_id, "full"),
				pool_length=pool_length,
				out_features=out_features,
				use_channel_pe=use_channel_pe,
			)

			checkpoint_dir = f"model_weights/finetuned/{experiment_name}"
			os.makedirs(checkpoint_dir, exist_ok=True)

			model_checkpoint = ModelCheckpoint(
				monitor="val_loss",
				dirpath=checkpoint_dir,
				filename=f"fold_{fold + 1}-best",
				save_top_k=1,
			)

			early_stopping = EarlyStopping(monitor="val_loss", patience=patience)

			trainer = Trainer(
				accelerator="auto",
				devices=torch.cuda.device_count() if device == "cuda" else 1,
				max_epochs=num_epochs,
				callbacks=[model_checkpoint, early_stopping],
				logger=wandb_logger,
				enable_progress_bar=True,
			)

			trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
			fold_result = trainer.test(model, dataloaders=test_loader, ckpt_path="best")
			logger.info(f"Fold {fold + 1} Test results: {fold_result}")
			all_fold_results.append(fold_result)

		if wandb.run is not None:
			wandb.finish()

		return all_fold_results