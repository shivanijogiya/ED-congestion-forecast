"""
Training pipeline entry point.

Usage:
    python -m graph_model.training.train_pipeline [--config config/model_config.yaml]
"""
import argparse
import logging
import torch
from torch.utils.data import DataLoader, random_split

from graph_model.model.ed_forecast_model import EDForecastModel
from graph_model.model.model_config import ModelConfig
from graph_model.training.dataset import EDForecastDataset
from graph_model.training.trainer import Trainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def collate_fn(batch):
    """Custom collate: return list of graph sequences + stacked targets."""
    graph_seqs = [item[0] for item in batch]
    targets    = torch.stack([item[1] for item in batch])
    num_nodes  = batch[0][2]
    return graph_seqs, targets, num_nodes


def run(config_path: str = None):
    config = ModelConfig.from_yaml(config_path) if config_path else ModelConfig()
    logger.info(f"Config: {config}")

    dataset = EDForecastDataset(config=config, demo_mode=True)
    logger.info(f"Dataset size: {len(dataset)} samples")

    # 80/20 train/val split
    val_size   = int(0.2 * len(dataset))
    train_size = len(dataset) - val_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        train_set, batch_size=config.batch_size,
        shuffle=True, collate_fn=collate_fn, num_workers=0,
    )
    val_loader = DataLoader(
        val_set, batch_size=config.batch_size,
        shuffle=False, collate_fn=collate_fn, num_workers=0,
    )

    model   = EDForecastModel(config)
    trainer = Trainer(model, config)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info("Starting training...")

    best_checkpoint = trainer.train(train_loader, val_loader)
    logger.info(f"Training complete. Best model: {best_checkpoint}")
    return best_checkpoint


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()
    run(args.config)
