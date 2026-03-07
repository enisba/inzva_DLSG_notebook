"""
Visualization utilities.
This file is FULLY PROVIDED — no changes needed.

Functions:
    - plot_dataset_exploration: Visualize feature/target distributions & correlations
    - plot_loss_curves: Plot train/test loss for a single run
    - plot_experiment_comparison: Overlay multiple experiments
    - plot_dataset_size_comparison: Compare train/test loss across dataset sizes
    - print_results_table: Print a formatted comparison table
"""

import os
import matplotlib.pyplot as plt
import numpy as np

# All plots are saved into this directory
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(RESULTS_DIR, exist_ok=True)


def _save_path(filename):
    """Return the full path inside the results/ directory."""
    return os.path.join(RESULTS_DIR, filename)


def plot_trajectories(simulate_fn):
    """
    Visualize the physics problem by plotting example projectile trajectories.
    Shows how each input parameter affects the flight path.

    Args:
        simulate_fn: function(v0, angle, mass, drag) -> (xs, ys)
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # --- 1. Varying initial velocity ---
    ax = axes[0, 0]
    for v0 in [15, 25, 35, 45]:
        xs, ys = simulate_fn(v0, 45, 1.0, 0.1)
        ax.plot(xs, ys, label=f"$v_0 = {v0}$ m/s", linewidth=2)
    ax.set_title(r"Varying Initial Velocity ($\theta=45°,\ m=1,\ C_d=0.1$)")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- 2. Varying launch angle ---
    ax = axes[0, 1]
    for angle in [15, 30, 45, 60, 75]:
        xs, ys = simulate_fn(30, angle, 1.0, 0.1)
        ax.plot(xs, ys, label=f"$\\theta = {angle}°$", linewidth=2)
    ax.set_title(r"Varying Launch Angle ($v_0=30,\ m=1,\ C_d=0.1$)")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- 3. Varying mass ---
    ax = axes[1, 0]
    for mass in [0.2, 0.5, 1.0, 3.0, 5.0]:
        xs, ys = simulate_fn(30, 45, mass, 0.1)
        ax.plot(xs, ys, label=f"$m = {mass}$ kg", linewidth=2)
    ax.set_title(r"Varying Mass ($v_0=30,\ \theta=45°,\ C_d=0.1$)")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    # --- 4. Varying drag coefficient ---
    ax = axes[1, 1]
    for cd in [0.0, 0.05, 0.1, 0.2, 0.4]:
        xs, ys = simulate_fn(30, 45, 1.0, cd)
        label = "No drag" if cd == 0.0 else f"$C_d = {cd}$"
        ax.plot(xs, ys, label=label, linewidth=2,
                linestyle="--" if cd == 0.0 else "-")
    ax.set_title(r"Varying Drag Coefficient ($v_0=30,\ \theta=45°,\ m=1$)")
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Height (m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(bottom=0)

    fig.suptitle("Projectile Motion with Air Drag — Example Trajectories",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    path = _save_path("trajectories.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")


def plot_dataset_exploration(X, y, feature_names=None):
    """
    Visualize the generated dataset:
      - Feature distributions (histograms)
      - Target distribution
      - Feature-target correlation scatter plots
      - Correlation heatmap

    Args:
        X: numpy array of shape (n_samples, n_features)
        y: numpy array of shape (n_samples,)
        feature_names: list of feature name strings
    """
    if feature_names is None:
        feature_names = ["Velocity (m/s)", "Angle (deg)", "Mass (kg)", "Drag Coeff"]

    n_features = X.shape[0] if X.ndim == 1 else X.shape[1]

    # --- 1. Feature distributions ---
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    for i in range(n_features):
        axes[i].hist(X[:, i], bins=40, color="#1f77b4", edgecolor="white", alpha=0.85)
        axes[i].set_title(feature_names[i])
        axes[i].set_xlabel("Value")
        axes[i].set_ylabel("Count")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Feature Distributions", fontsize=14, y=1.02)
    plt.tight_layout()
    path = _save_path("dataset_feature_distributions.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")

    # --- 2. Target distribution ---
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(y, bins=50, color="#2ca02c", edgecolor="white", alpha=0.85)
    ax.set_xlabel("Horizontal Distance (m)")
    ax.set_ylabel("Count")
    ax.set_title("Target Distribution — Horizontal Distance")
    ax.axvline(y.mean(), color="red", linestyle="--", label=f"Mean = {y.mean():.1f} m")
    ax.axvline(np.median(y), color="orange", linestyle="--", label=f"Median = {np.median(y):.1f} m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = _save_path("dataset_target_distribution.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")

    # --- 3. Feature vs Target scatter plots ---
    fig, axes = plt.subplots(1, n_features, figsize=(4 * n_features, 4))
    for i in range(n_features):
        axes[i].scatter(X[:, i], y, alpha=0.3, s=8, color="#d62728")
        axes[i].set_xlabel(feature_names[i])
        axes[i].set_ylabel("Distance (m)")
        axes[i].set_title(f"{feature_names[i]} vs Distance")
        axes[i].grid(True, alpha=0.3)
    fig.suptitle("Feature–Target Relationships", fontsize=14, y=1.02)
    plt.tight_layout()
    path = _save_path("dataset_feature_vs_target.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")

    # --- 4. Correlation heatmap ---
    data = np.column_stack([X, y])
    labels = feature_names + ["Distance"]
    corr = np.corrcoef(data, rowvar=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{corr[i, j]:.2f}", ha="center", va="center",
                    fontsize=10, color="black" if abs(corr[i, j]) < 0.6 else "white")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    path = _save_path("dataset_correlation_matrix.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")

    # --- 5. Print summary statistics ---
    print(f"\n  Dataset Summary ({X.shape[0]} samples)")
    print(f"  {'-' * 55}")
    print(f"  {'Feature':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-' * 55}")
    for i, name in enumerate(feature_names):
        print(f"  {name:<20} {X[:, i].mean():>10.2f} {X[:, i].std():>10.2f} "
              f"{X[:, i].min():>10.2f} {X[:, i].max():>10.2f}")
    print(f"  {'Distance (target)':<20} {y.mean():>10.2f} {y.std():>10.2f} "
          f"{y.min():>10.2f} {y.max():>10.2f}")
    print(f"  {'-' * 55}")


def plot_loss_curves(history, title="Training Progress"):
    """
    Plot training and test loss curves for a single experiment.

    Args:
        history: dict with 'train_loss' and 'test_loss' lists
        title: Plot title
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(10, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss", color="blue")
    plt.plot(epochs, history["test_loss"], label="Test Loss",
             color="red", linestyle="--")
    plt.xlabel("Epoch")
    plt.ylabel("Loss (MSE)")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    path = _save_path(f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")


def plot_experiment_comparison(results, title="Experiment Comparison"):
    """
    Overlay training and test loss curves from multiple experiments.

    Args:
        results: dict mapping experiment names to history dicts
                 e.g. {"Adam": {"train_loss": [...], "test_loss": [...]}, ...}
        title: Plot title
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for i, (name, history) in enumerate(results.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        color = colors[i % len(colors)]

        # Training loss
        axes[0].plot(epochs, history["train_loss"], label=name, color=color)
        # Test loss
        axes[1].plot(epochs, history["test_loss"], label=name,
                     color=color, linestyle="--")

    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title(f"{title} — Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].set_title(f"{title} — Test Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    path = _save_path(f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")


def plot_dataset_size_comparison(results, title="Dataset Size Comparison"):
    """
    Compare training and test loss across different dataset sizes.
    Includes a bar chart of final train vs test loss.

    Args:
        results: dict mapping size names to history dicts
        title: Plot title
    """
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Plot 1: Training loss curves
    for i, (name, history) in enumerate(results.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[0].plot(epochs, history["train_loss"], label=name,
                     color=colors[i % len(colors)])
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (MSE)")
    axes[0].set_title("Training Loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Test loss curves
    for i, (name, history) in enumerate(results.items()):
        epochs = range(1, len(history["train_loss"]) + 1)
        axes[1].plot(epochs, history["test_loss"], label=name,
                     color=colors[i % len(colors)], linestyle="--")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss (MSE)")
    axes[1].set_title("Test Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Bar chart of final losses
    names = list(results.keys())
    final_train = [results[n]["train_loss"][-1] for n in names]
    final_test = [results[n]["test_loss"][-1] for n in names]

    x = np.arange(len(names))
    width = 0.35
    axes[2].bar(x - width / 2, final_train, width, label="Train", color="#4CAF50")
    axes[2].bar(x + width / 2, final_test, width, label="Test", color="#F44336")
    axes[2].set_xlabel("Dataset Size")
    axes[2].set_ylabel("Final Loss (MSE)")
    axes[2].set_title("Final Train vs Test Loss")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=15)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    path = _save_path(f"{title.lower().replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.show()
    print(f"  Plot saved: {path}")


def print_results_table(results, title="Results"):
    """
    Print a formatted table comparing final loss values across experiments.

    Args:
        results: dict mapping experiment names to history dicts
        title: Table title
    """
    print(f"\n{'=' * 65}")
    print(f"  {title} — Summary")
    print(f"{'=' * 65}")
    print(f"  {'Configuration':<25} {'Train MSE':>12} {'Test MSE':>12} {'Gap':>10}")
    print(f"  {'-' * 60}")

    for name, history in results.items():
        train_final = history["train_loss"][-1]
        test_final = history["test_loss"][-1]
        gap = test_final - train_final
        print(f"  {name:<25} {train_final:>12.6f} {test_final:>12.6f} {gap:>+10.6f}")

    print(f"{'=' * 65}")
