"""
Evaluation & experiment runner module for Assignment 2.
Students must complete the experiment functions.
"""

import torch
import torch.nn as nn
import numpy as np

from config import (SEED, TRAIN_CONFIG, MODEL_CONFIG,
                    OPTIMIZER_EXPERIMENT, REGULARIZATION_EXPERIMENT,
                    DATASET_SIZE_EXPERIMENT, DATA_CONFIG)
from data import prepare_data
from model import create_model
from train import train_model, get_optimizer
from visualize import (plot_loss_curves, plot_experiment_comparison,
                       plot_dataset_size_comparison, print_results_table)


def set_seed(seed=SEED):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)


def run_single_experiment(model_type, optimizer_name, train_loader, test_loader,
                          lr, epochs, weight_decay=0.0, momentum=0.9, seed=SEED):
    """
    Train a single model configuration and return the training history.

    Args:
        model_type: "standard" or "dropout"
        optimizer_name: "adam" or "sgd"
        train_loader: Training DataLoader
        test_loader: Test DataLoader
        lr: Learning rate
        epochs: Number of epochs
        weight_decay: L2 regularization strength
        momentum: SGD momentum
        seed: Random seed

    Returns:
        history: dict with 'train_loss' and 'test_loss' lists
        model: The trained model
    """
    set_seed(seed)

    model = create_model(model_type)
    optimizer = get_optimizer(model, optimizer_name, lr=lr, momentum=momentum, weight_decay=weight_decay)
    criterion = nn.MSELoss()
    history = train_model(model, train_loader, test_loader, criterion, optimizer, epochs)

    return history, model



def experiment_optimizers():
    """
    Compare Adam vs SGD+Momentum on the same dataset.

    Steps:
        1. Prepare a medium-sized dataset
        2. Train with Adam (lr from config)
        3. Train with SGD+Momentum (lr from config)
        4. Plot and compare results
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 1: Optimizer Comparison (Adam vs SGD+Momentum)")
    print("=" * 60)

    train_loader, test_loader, _ = prepare_data(DATA_CONFIG["n_samples_medium"])
    epochs = OPTIMIZER_EXPERIMENT["epochs"]

    adam_h, _ = run_single_experiment("standard", "adam", train_loader, test_loader, lr=TRAIN_CONFIG["learning_rate"], epochs=epochs)

    sgd_h, _ = run_single_experiment("standard", "sgd", train_loader, test_loader, lr=TRAIN_CONFIG["sgd_learning_rate"], momentum=TRAIN_CONFIG["sgd_momentum"], epochs=epochs)

    results = {
        "Adam": adam_h,
        "SGD+Momentum": sgd_h}

    plot_experiment_comparison(results, "Optimizer Comparison Adam vs SGD+Momentum")
    print_results_table(results)
    return results


def experiment_regularization():
    """
    Compare No Regularization vs Dropout vs Weight Decay.

    Steps:
        1. Prepare a medium-sized dataset
        2. Train with no regularization (standard model, weight_decay=0)
        3. Train with Dropout (dropout model, weight_decay=0)
        4. Train with Weight Decay (standard model, weight_decay > 0)
        5. Plot and compare results
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 2: Regularization Comparison")
    print("=" * 60)

    epochs = REGULARIZATION_EXPERIMENT["epochs"]
    train_loader, test_loader, _ = prepare_data(DATA_CONFIG["n_samples_medium"])

    h_none, _ = run_single_experiment("standard", "adam", train_loader, test_loader, lr=TRAIN_CONFIG["learning_rate"], epochs=epochs)

    h_dropout, _ = run_single_experiment("dropout", "adam", train_loader, test_loader, lr=TRAIN_CONFIG["learning_rate"], epochs=epochs)

    h_weight_decay, _ = run_single_experiment("standard", "adam", train_loader, test_loader, lr=TRAIN_CONFIG["learning_rate"], weight_decay=TRAIN_CONFIG["weight_decay"], epochs=epochs)
    
    results = {"No Regularization": h_none, "Dropout": h_dropout, "Weight Decay": h_weight_decay}  
    plot_experiment_comparison(results, "Regularization Comparison None vs Dropout vs Weight Decay")
    print_results_table(results)
    return results

def experiment_dataset_size():
    """
    Compare model performance with Small vs Medium vs Large datasets.

    Steps:
        1. Generate 3 datasets of different sizes
        2. Train the same model architecture on each
        3. Plot and compare results
    """
    print("\n" + "=" * 60)
    print("  EXPERIMENT 3: Dataset Size Comparison")
    print("=" * 60)

    epochs = DATASET_SIZE_EXPERIMENT["epochs"]

    sizes = {
        "Small": DATA_CONFIG["n_samples_small"],
        "Medium": DATA_CONFIG["n_samples_medium"],
        "Large": DATA_CONFIG["n_samples_large"]
    }
    results = {}

    for name, count in sizes.items():
        train_loader, test_loader, _ = prepare_data(count)
        history, _ = run_single_experiment("standard", "adam", train_loader, test_loader, lr=TRAIN_CONFIG["learning_rate"], epochs=DATASET_SIZE_EXPERIMENT["epochs"])
        results[name] = history

    plot_dataset_size_comparison(results, "Dataset Size Comparison Small vs Medium vs Large")
    print_results_table(results)
    return results
