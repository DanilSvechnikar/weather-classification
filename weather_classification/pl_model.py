"""This module contains classes related to pytorch lightning model."""

from pathlib import Path

import lightning as pl
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim

from .model_utils import load_weights_lt_model
from .types.enums import MetricNames


class LiEfficientNet(pl.LightningModule):
    """PyTorch Lighting model."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: optim.Optimizer,
        loss_fn: nn.Module,
        metrics,
        lr_scheduler,
        class_labels,
    ):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.metrics = metrics
        self.lr_scheduler = lr_scheduler
        self.learning_rate = optimizer.param_groups[0]["lr"]
        self.class_labels = class_labels

        self.test_preds: list[float] = []
        self.test_labels: list[float] = []

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self.model(x)
        preds = torch.argmax(logits, dim=1)
        loss = self.loss_fn(logits, y)
        return loss, preds, y

    def training_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        accuracy = self.metrics["accuracy"](preds, y)

        self.log(
            MetricNames.train_loss.value,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            MetricNames.train_accuracy.value,
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        return loss

    def validation_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        accuracy = self.metrics["accuracy"](preds, y)
        precision = self.metrics["precision"](preds, y)

        self.log(
            MetricNames.val_loss.value,
            loss,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            MetricNames.val_accuracy.value,
            accuracy,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )
        self.log(
            MetricNames.val_precision.value,
            precision,
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        loss, preds, y = self._shared_step(batch)
        self.test_preds.append(preds)
        self.test_labels.append(y)

        accuracy = self.metrics["accuracy"](preds, y)
        precision = self.metrics["precision"](preds, y)

        self.log_dict(
            {
                MetricNames.test_loss.value: loss,
                MetricNames.test_accuracy.value: accuracy,
                MetricNames.test_precision.value: precision,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def on_test_epoch_end(self) -> None:
        test_preds = torch.cat(self.test_preds)
        test_labels = torch.cat(self.test_labels)

        cm = self.metrics["conf_matrix"](test_labels, test_preds)

        self.plot_confusion_matrix(cm)
        self.test_preds.clear()
        self.test_labels.clear()

    def plot_confusion_matrix(self, cm):
        """Save confusion matrix as png and to logger."""
        fig, ax = plt.subplots(figsize=(10, 10))
        sns.heatmap(
            cm.cpu(),
            annot=True,
            fmt=".1%",
            linewidth=0.1,
            cmap="GnBu",
        )
        ax.set_xlabel("Pred")
        ax.set_ylabel("True")
        ax.set_title("Confusion Matrix")

        cm_path = Path(self.trainer.log_dir) / "Confusion Matrix.png"
        fig.savefig(cm_path, bbox_inches="tight")
        self.logger.experiment.add_figure("Confusion Matrix", fig)
        plt.close(fig)

    def configure_optimizers(self) -> dict:
        optimizer = self.optimizer

        if self.lr_scheduler:
            lr_scheduler_config = {
                "scheduler": self.lr_scheduler,
                "interval": "epoch",
                "monitor": MetricNames.val_loss.value,
            }
            return {
                "optimizer": optimizer,
                "lr_scheduler": lr_scheduler_config,
            }

        return {"optimizer": optimizer}

    def load_model_weights(self, model_fpath: Path | str) -> None:
        """Loads weights into the model."""
        model_weights = load_weights_lt_model(model_fpath)
        self.model.load_state_dict(model_weights)
