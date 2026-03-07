"""
Configuration file for the assignment.
All hyperparameters and experiment settings are defined here.

Modify these values to run different experiments.
Do NOT hardcode hyperparameters in other files — import them from here.
"""

SEED = 42

DATA_CONFIG = {
    "n_samples_small": 500,
    "n_samples_medium": 2000,
    "n_samples_large": 10000,
    "test_ratio": 0.2,
    # Physics parameter ranges
    "velocity_range": (10.0, 50.0),     # m/s
    "angle_range": (5.0, 80.0),          # degrees
    "mass_range": (0.1, 5.0),            # kg
    "drag_coeff_range": (0.01, 0.5),     # dimensionless
}

MODEL_CONFIG = {
    "input_dim": 4,          # (velocity, angle, mass, drag_coeff)
    "hidden_dims": [16, 16, 8],  # Hidden layer sizes
    "output_dim": 1,         # distance
    "dropout_rate": 0.2,     # Used only when dropout is enabled
}

TRAIN_CONFIG = {
    "epochs": 200,
    "batch_size": 16,
    "learning_rate": 5e-4,
    "sgd_learning_rate": 1e-2,
    "sgd_momentum": 0.9,
    "weight_decay": 1e-4,    # L2 regularization strength
}

# Experiment 1
OPTIMIZER_EXPERIMENT = {
    "optimizers": ["adam", "sgd", 'adamw'],
    "epochs": 200,
    "dataset_size": "medium",  # Use medium dataset
}

# Experiment 2
REGULARIZATION_EXPERIMENT = {
    "methods": ["none", "dropout"],
    "epochs": 200,
    "dataset_size": "medium",
}

# Experiment 3
DATASET_SIZE_EXPERIMENT = {
    "sizes": ["small", "medium", "large"],
    "epochs": 200,
    "optimizer": "adam",
}
