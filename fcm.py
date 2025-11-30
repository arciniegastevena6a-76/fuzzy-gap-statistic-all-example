"""
Fuzzy C-Means clustering implementation
With convergence checking and robustness improvements
"""

import numpy as np

# Numerical constants
DISTANCE_MIN_VALUE = 1e-10  # Minimum value to avoid division by zero
DEGENERATE_CLUSTER_THRESHOLD = 1e-6  # Threshold for detecting empty clusters


class FuzzyCMeans:
    """
    Fuzzy C-Means clustering algorithm
    
    According to paper Section 2.3 and Eq.(7):
    J_m = Σ_i Σ_j (u_ij)^m * d
    
    With convergence checking and degenerate cluster handling.
    """

    def __init__(self, n_clusters: int = 3, m: float = 2.0,
                 max_iter: int = 100, error: float = 1e-5,
                 random_seed: int = None):
        """
        Initialize FCM

        Args:
            n_clusters: Number of clusters
            m: Fuzziness parameter (default 2.0)
            max_iter: Maximum iterations
            error: Convergence threshold
            random_seed: Random seed for reproducibility
        """
        self.n_clusters = n_clusters
        self.m = m
        self.max_iter = max_iter
        self.error = error
        self.random_seed = random_seed

        self.centers = None
        self.u = None  # Membership matrix
        self.objective_value = None
        self.converged = False
        self.n_iterations = 0

    def _initialize_membership(self, n_samples: int) -> np.ndarray:
        """
        Initialize membership matrix randomly
        """
        if self.random_seed is not None:
            rng = np.random.RandomState(self.random_seed)
            u = rng.rand(n_samples, self.n_clusters)
        else:
            u = np.random.rand(n_samples, self.n_clusters)
        u = u / np.sum(u, axis=1, keepdims=True)
        return u

    def _update_centers(self, X: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Update cluster centers
        """
        um = u ** self.m
        denominator = np.sum(um.T, axis=1, keepdims=True)
        # Avoid division by zero (degenerate clusters)
        denominator = np.maximum(denominator, DISTANCE_MIN_VALUE)
        centers = (um.T @ X) / denominator
        return centers

    def _update_membership(self, X: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Update membership matrix
        """
        n_samples = X.shape[0]
        u = np.zeros((n_samples, self.n_clusters))

        for i in range(n_samples):
            for j in range(self.n_clusters):
                distances = np.linalg.norm(X[i] - centers, axis=1)
                distances[distances == 0] = DISTANCE_MIN_VALUE  # Avoid division by zero

                u[i, j] = 1.0 / np.sum((distances[j] / distances) ** (2.0 / (self.m - 1)))

        return u

    def _calculate_objective(self, X: np.ndarray, u: np.ndarray,
                            centers: np.ndarray) -> float:
        """
        Calculate objective function value
        Paper Eq.(7): J_m = Σ_i Σ_j (u_ij)^m * d
        """
        um = u ** self.m
        distances = np.linalg.norm(X[:, np.newaxis] - centers, axis=2)
        obj = np.sum(um * (distances ** 2))
        return obj

    def _check_degenerate_clusters(self, u: np.ndarray) -> bool:
        """
        Check if any cluster has become degenerate (no samples assigned)
        
        Returns:
            True if clusters are degenerate
        """
        cluster_membership = np.sum(u, axis=0)
        return np.any(cluster_membership < DEGENERATE_CLUSTER_THRESHOLD)

    def fit(self, X: np.ndarray):
        """
        Fit FCM to data

        Args:
            X: Input data (n_samples, n_features)
        """
        n_samples, n_features = X.shape

        # Initialize membership matrix
        self.u = self._initialize_membership(n_samples)
        self.converged = False
        self.n_iterations = 0

        for iteration in range(self.max_iter):
            u_old = self.u.copy()

            # Update centers
            self.centers = self._update_centers(X, self.u)

            # Update membership
            self.u = self._update_membership(X, self.centers)

            self.n_iterations = iteration + 1

            # Check convergence
            diff = np.linalg.norm(self.u - u_old)
            if diff < self.error:
                self.converged = True
                break

            # Check for degenerate clusters
            if self._check_degenerate_clusters(self.u):
                # Reinitialize if clusters degenerate
                self.u = self._initialize_membership(n_samples)

        # Calculate final objective value
        self.objective_value = self._calculate_objective(X, self.u, self.centers)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels

        Args:
            X: Input data

        Returns:
            labels: Cluster labels
        """
        u = self._update_membership(X, self.centers)
        labels = np.argmax(u, axis=1)
        return labels
    
    def get_membership(self, X: np.ndarray = None) -> np.ndarray:
        """
        Get membership matrix
        
        Args:
            X: Input data (if None, returns membership from fit)
            
        Returns:
            u: Membership matrix (n_samples, n_clusters)
        """
        if X is None:
            return self.u
        return self._update_membership(X, self.centers)