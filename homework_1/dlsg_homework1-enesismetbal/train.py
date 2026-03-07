"""
Training module
Students must complete the training and validation loops.
"""

import torch
import torch.nn as nn
import numpy as np


def train_one_epoch(model, train_loader, criterion, optimizer):
    """
    Train the model for one epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer

    Returns:
        avg_loss: Average training loss for this epoch
    """
    model.train()
    total_loss = 0.0
    num_batches = 0

    for X_batch, y_batch in train_loader:
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def validate(model, test_loader, criterion):
    """
    Evaluate the model on the test/validation set.

    Args:
        model: The neural network model
        test_loader: DataLoader for test data
        criterion: Loss function

    Returns:
        avg_loss: Average test loss
    """
    model.eval()
    total_loss = 0.0
    num_batches = 0

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            total_loss += loss.item()
            num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    return avg_loss


def train_model(model, train_loader, test_loader, criterion, optimizer,
                epochs=100, verbose=True):
    """
    Full training loop: train for multiple epochs, tracking both
    training and test loss at each epoch.

    Args:
        model: The neural network model
        train_loader: DataLoader for training data
        test_loader: DataLoader for test data
        criterion: Loss function
        optimizer: Optimizer
        epochs: Number of training epochs
        verbose: Whether to print progress

    Returns:
        history: dict with keys 'train_loss' and 'test_loss',
                 each containing a list of per-epoch loss values

    """
    history = {"train_loss": [], "test_loss": []}

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer)
        test_loss = validate(model, test_loader, criterion)

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)

        if verbose and (epoch + 1)%20 == 0:
            print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}, Test Loss: {test_loss:.4f}")

    return history


def get_optimizer(model, optimizer_name="adam", lr=1e-3, momentum=0.9,
                  weight_decay=0.0):
    """
    Create an optimizer for the given model.

    Args:
        model: The neural network model
        optimizer_name: "adam" or "sgd"
        lr: Learning rate
        momentum: Momentum (only used for SGD)
        weight_decay: L2 regularization strength (0.0 = no weight decay)

    Returns:
        optimizer: A PyTorch optimizer instance
    """

    if optimizer_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        return torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Invalid optimizer_name '{optimizer_name}'. Expected 'adam', 'sgd', or 'adamw'.")