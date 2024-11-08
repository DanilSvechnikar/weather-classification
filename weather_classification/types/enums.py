"""This file contains enums."""

from enum import Enum


class Stage(Enum):
    """This enumeration contains stages."""

    train = "train"
    val = "val"
    test = "test"


class MetricNames(Enum):
    """This enumeration contains metric names."""

    train_loss = f"{Stage.train.value}_loss"
    train_accuracy = f"{Stage.train.value}_acc"

    val_loss = f"{Stage.val.value}_loss"
    val_accuracy = f"{Stage.val.value}_acc"
    val_precision = f"{Stage.val.value}_precision"

    test_loss = f"{Stage.test.value}_loss"
    test_accuracy = f"{Stage.test.value}_acc"
    test_precision = f"{Stage.test.value}_precision"

    Loss = "Loss"
    Metrics = "Metrics"
    Test = "Test"
