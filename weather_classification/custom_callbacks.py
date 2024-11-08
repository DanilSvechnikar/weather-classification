"""This file contains custom callbacks implementations."""

from lightning.pytorch.callbacks import TQDMProgressBar
from lightning.pytorch.loggers import TensorBoardLogger

from .types.enums import MetricNames, Stage


class CustomTensorBoardLogger(TensorBoardLogger):
    """Custom TensorBoard logger."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._trainer = None

    @property
    def trainer(self):
        return self._trainer

    @trainer.setter
    def trainer(self, trainer):
        self._trainer = trainer

    def _process_metrics(self, metrics, metric_type):
        """Helper function to group metrics."""
        metric_mappings = {
            MetricNames.train_loss.value: f"{MetricNames.Loss.value}/{MetricNames.train_loss.value}",
            MetricNames.train_accuracy.value: f"{MetricNames.Metrics.value}/{MetricNames.train_accuracy.value}",
            MetricNames.val_loss.value: f"{MetricNames.Loss.value}/{MetricNames.val_loss.value}",
            MetricNames.val_accuracy.value: f"{MetricNames.Metrics.value}/{MetricNames.val_accuracy.value}",
            MetricNames.val_precision.value: f"{MetricNames.Metrics.value}/{MetricNames.val_precision.value}",
            MetricNames.test_loss.value: f"{MetricNames.Test.value}/{MetricNames.test_loss.value}",
            MetricNames.test_accuracy.value: f"{MetricNames.Test.value}/{MetricNames.test_accuracy.value}",
            MetricNames.test_precision.value: f"{MetricNames.Test.value}/{MetricNames.test_precision.value}",
        }

        for metric_name, new_metric_name in metric_mappings.items():
            if metric_name in metrics and metric_name.startswith(metric_type):
                metrics[new_metric_name] = metrics.pop(metric_name)

        return metrics

    def log_metrics(self, metrics, step):
        step = self.trainer.current_epoch

        if self.trainer.state.stage.TRAINING:
            metrics = self._process_metrics(metrics, Stage.train.value)

        if self.trainer.state.stage.VALIDATING:
            metrics = self._process_metrics(metrics, Stage.val.value)
            metrics.pop("epoch", None)

        if self.trainer.state.stage.TESTING:
            metrics = self._process_metrics(metrics, Stage.test.value)
            metrics.pop("epoch", None)

        super().log_metrics(metrics, step)


class CustomTQDMProgressBar(TQDMProgressBar):
    """Custom TQDM Progress Bar."""

    def get_metrics(self, trainer, pl_module):
        items = super().get_metrics(trainer, pl_module)
        items.pop("v_num", None)
        return items
