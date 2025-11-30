"""
Gap Statistic calculator for determining optimal number of clusters
"""

import numpy as np
from typing import Dict


class GapStatisticCalculator:
    """
    Calculate Fuzzy Gap Statistic (FGS)
    """

    def __init__(self, max_iterations: int = 100):
        """
        Initialize Gap Statistic Calculator

        Args:
            max_iterations: Maximum iterations for FCM
        """
        self.max_iterations = max_iterations
        self.s_k = {}  # Standard deviation for each k

    def calculate_gap(self, log_jmk_star_list: list,
                     log_jmk: float, k: int) -> float:
        """
        Calculate Gap statistic for a given k

        Args:
            log_jmk_star_list: List of log objective values from Monte Carlo samples
            log_jmk: Log objective value from original data
            k: Number of clusters

        Returns:
            gap_k: Gap statistic value
        """
        # Calculate mean of log(J_mk*)
        mean_log_jmk_star = np.mean(log_jmk_star_list)

        # Calculate Gap(k)
        gap_k = mean_log_jmk_star - log_jmk

        # Calculate standard deviation
        std_log_jmk_star = np.std(log_jmk_star_list)
        self.s_k[k] = std_log_jmk_star * np.sqrt(1 + 1.0 / len(log_jmk_star_list))

        return gap_k

    def find_optimal_k(self, fgs_results: Dict[int, float]) -> int:
        """
        Find optimal k using Gap statistic criterion

        Criterion: Gap(k) >= Gap(k+1) - s_{k+1}

        Args:
            fgs_results: Dictionary of {k: Gap(k)}

        Returns:
            optimal_k: Optimal number of clusters
        """
        optimal_k = 1

        for k in sorted(fgs_results.keys())[:-1]:
            gap_k = fgs_results[k]
            gap_k_plus_1 = fgs_results[k + 1]
            s_k_plus_1 = self.s_k.get(k + 1, 0.0)

            # Check criterion
            if gap_k >= gap_k_plus_1 - s_k_plus_1:
                optimal_k = k
                break
        else:
            # If no k satisfies criterion, choose the one with maximum Gap
            optimal_k = max(fgs_results, key=fgs_results.get)

        return optimal_k