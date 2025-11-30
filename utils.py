"""
Utility functions for data preprocessing
"""

import numpy as np


def standardize_data(data: np.ndarray) -> np.ndarray:
    """
    Standardize data to zero mean and unit variance

    Args:
        data: Input data (n_samples, n_features)

    Returns:
        standardized_data: Standardized data
    """
    mean = np.mean(data, axis=0)
    std = np.std(data, axis=0)

    # Avoid division by zero
    std[std == 0] = 1.0

    standardized_data = (data - mean) / std

    return standardized_data