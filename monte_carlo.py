"""
Monte Carlo sampling for Fuzzy Gap Statistic
With random seed control for reproducibility (Paper Algorithm 2)
"""

import numpy as np


class MonteCarloSampling:
    """
    Monte Carlo sampling class with random seed support
    
    According to the paper Algorithm 2, Monte Carlo sampling is used
    to generate reference distributions for FGS calculation.
    Typically 20 samples are generated (B=20 in paper).
    """

    def __init__(self, random_seed: int = None):
        """
        Initialize Monte Carlo sampler
        
        Args:
            random_seed: Random seed for reproducibility. If None, results
                        will not be reproducible across runs.
        """
        self.random_seed = random_seed
        self._rng = None
        if random_seed is not None:
            self._rng = np.random.RandomState(random_seed)

    def set_seed(self, seed: int):
        """
        Set random seed for reproducibility
        
        Args:
            seed: Random seed value
        """
        self.random_seed = seed
        self._rng = np.random.RandomState(seed)

    def _generate_uniform_samples(self, mins: np.ndarray, maxs: np.ndarray, 
                                   size: tuple) -> np.ndarray:
        """
        Generate uniform samples using the appropriate random generator
        
        Args:
            mins: Minimum values for each feature
            maxs: Maximum values for each feature
            size: Output shape (n_samples, n_features)
            
        Returns:
            sampled_data: Uniformly sampled data
        """
        if self._rng is not None:
            return self._rng.uniform(low=mins, high=maxs, size=size)
        return np.random.uniform(low=mins, high=maxs, size=size)

    def sample_uniform(self, data: np.ndarray) -> np.ndarray:
        """
        Sample uniformly from the bounding box of the data
        
        According to paper Step 2.1: Monte Carlo sampling is carried out
        for each attribute according to the uniform distribution on the
        interval [min_ij, max_ij].

        Args:
            data: Original data (n_samples, n_features)

        Returns:
            sampled_data: Sampled data with same shape
        """
        n_samples, n_features = data.shape

        # Get min and max for each feature
        mins = np.min(data, axis=0)
        maxs = np.max(data, axis=0)

        # Sample uniformly using the appropriate random generator
        return self._generate_uniform_samples(mins, maxs, (n_samples, n_features))