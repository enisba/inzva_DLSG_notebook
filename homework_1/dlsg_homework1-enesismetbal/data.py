"""
Data generation module
Simulates projectile motion with quadratic air drag using numerical integration.

The physics:
    A projectile is launched with initial velocity v0 at angle theta.

    Drag force (opposes velocity, proportional to v^2):
        F_drag = -C_d * |v| * v

    Gravitational force:
        F_gravity = -m * g * y_hat

    Component-wise accelerations:
        a_x = -(C_d / m) * |v| * v_x
        a_y = -g - (C_d / m) * |v| * v_y

    where |v| = sqrt(v_x^2 + v_y^2).

    We integrate using Euler's method with a small time step (dt) to
    compute the horizontal distance traveled before the projectile
    hits the ground.

This file is PROVIDED — no changes needed.
"""

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from config import DATA_CONFIG, SEED


def simulate_projectile(v0, angle_deg, mass, drag_coeff, dt=0.001, g=9.81):
    """
    Simulate a projectile with quadratic air drag and return the
    horizontal distance traveled.

    Args:
        v0: Initial speed (m/s)
        angle_deg: Launch angle in degrees
        mass: Mass of the projectile (kg)
        drag_coeff: Drag coefficient (dimensionless)
        dt: Time step for Euler integration (s)
        g: Gravitational acceleration (m/s^2)

    Returns:
        distance: Horizontal distance traveled (m)
    """
    angle_rad = np.radians(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x = 0.0
    y = 0.0

    # Euler integration until projectile hits ground (y < 0)
    max_steps = 100000  # Safety limit
    for _ in range(max_steps):
        speed = np.sqrt(vx**2 + vy**2)

        # Drag force (opposes velocity, proportional to v^2)
        drag_x = -drag_coeff * speed * vx / mass
        drag_y = -drag_coeff * speed * vy / mass

        # Update velocity
        vx += drag_x * dt
        vy += (-g + drag_y) * dt

        # Update position
        x += vx * dt
        y += vy * dt

        # Stop when projectile hits the ground
        if y < 0:
            break

    return max(x, 0.0)  # Distance can't be negative


def simulate_projectile_trajectory(v0, angle_deg, mass, drag_coeff, dt=0.001, g=9.81):
    """
    Simulate a projectile and return the full (x, y) trajectory.

    Returns:
        xs: list of x positions
        ys: list of y positions
    """
    angle_rad = np.radians(angle_deg)
    vx = v0 * np.cos(angle_rad)
    vy = v0 * np.sin(angle_rad)
    x, y = 0.0, 0.0
    xs, ys = [x], [y]

    max_steps = 100000
    record_every = max(1, int(0.01 / dt))  # Record every ~0.01s
    for step in range(max_steps):
        speed = np.sqrt(vx**2 + vy**2)
        drag_x = -drag_coeff * speed * vx / mass
        drag_y = -drag_coeff * speed * vy / mass
        vx += drag_x * dt
        vy += (-g + drag_y) * dt
        x += vx * dt
        y += vy * dt
        if step % record_every == 0:
            xs.append(x)
            ys.append(max(y, 0.0))
        if y < 0:
            xs.append(x)
            ys.append(0.0)
            break

    return xs, ys


def generate_dataset(n_samples, seed=SEED):
    """
    Generate a dataset of projectile simulations.

    Args:
        n_samples: Number of data points to generate
        seed: Random seed for reproducibility

    Returns:
        X: Input features array (n_samples, 4)
           Columns: [velocity, angle, mass, drag_coeff]
        y: Target array (n_samples,)
           The horizontal distance traveled
    """
    rng = np.random.RandomState(seed)

    # Sample input parameters uniformly from their ranges
    velocities = rng.uniform(*DATA_CONFIG["velocity_range"], size=n_samples)
    angles = rng.uniform(*DATA_CONFIG["angle_range"], size=n_samples)
    masses = rng.uniform(*DATA_CONFIG["mass_range"], size=n_samples)
    drag_coeffs = rng.uniform(*DATA_CONFIG["drag_coeff_range"], size=n_samples)

    # Compute distances via simulation
    distances = np.array([
        simulate_projectile(v, a, m, cd)
        for v, a, m, cd in zip(velocities, angles, masses, drag_coeffs)
    ])

    X = np.column_stack([velocities, angles, masses, drag_coeffs])
    y = distances

    return X, y


def normalize_data(X_train, X_test, y_train, y_test):
    """
    Normalize features and targets to zero mean, unit variance.
    Fit statistics on training data ONLY, then apply to both sets.

    Args:
        X_train, X_test: Feature arrays
        y_train, y_test: Target arrays

    Returns:
        X_train_norm, X_test_norm: Normalized feature arrays
        y_train_norm, y_test_norm: Normalized target arrays
        stats: dict with mean/std for features and targets (for denormalization)
    """
    # Feature normalization
    X_mean = X_train.mean(axis=0)
    X_std = X_train.std(axis=0)
    X_train_norm = (X_train - X_mean) / X_std
    X_test_norm = (X_test - X_mean) / X_std

    # Target normalization
    y_mean = y_train.mean()
    y_std = y_train.std()
    y_train_norm = (y_train - y_mean) / y_std
    y_test_norm = (y_test - y_mean) / y_std

    stats = {
        "X_mean": X_mean, "X_std": X_std,
        "y_mean": y_mean, "y_std": y_std,
    }

    return X_train_norm, X_test_norm, y_train_norm, y_test_norm, stats


def create_dataloaders(X_train, X_test, y_train, y_test, batch_size=64):
    """
    Convert numpy arrays to PyTorch DataLoaders.

    Args:
        X_train, X_test: Feature arrays (numpy)
        y_train, y_test: Target arrays (numpy)
        batch_size: Batch size for DataLoader

    Returns:
        train_loader, test_loader: PyTorch DataLoaders
    """
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    test_dataset = TensorDataset(X_test_t, y_test_t)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader


def prepare_data(n_samples, batch_size=64, seed=SEED):
    """
    End-to-end data pipeline: generate → split → normalize → DataLoaders.

    Args:
        n_samples: Number of total samples to generate
        batch_size: Batch size for DataLoaders
        seed: Random seed

    Returns:
        train_loader, test_loader: PyTorch DataLoaders
        stats: Normalization statistics for denormalization
    """
    from sklearn.model_selection import train_test_split

    print(f"Generating {n_samples} projectile simulations...")
    X, y = generate_dataset(n_samples, seed=seed)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=DATA_CONFIG["test_ratio"], random_state=seed
    )
    print(f"  Train: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")

    X_train_n, X_test_n, y_train_n, y_test_n, stats = normalize_data(
        X_train, X_test, y_train, y_test
    )
    print(f"  Features normalized (zero mean, unit variance)")

    train_loader, test_loader = create_dataloaders(
        X_train_n, X_test_n, y_train_n, y_test_n, batch_size=batch_size
    )

    return train_loader, test_loader, stats
