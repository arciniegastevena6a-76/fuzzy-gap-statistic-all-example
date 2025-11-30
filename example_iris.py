"""
Iris dataset experiment - Simplified output format
基于论文 Section 4.1 的 Iris 数据集实验
"""

import numpy as np
from sklearn.datasets import load_iris
from fuzzy_gap_statistic import FuzzyGapStatistic

# Paper reference values
PAPER_M_EMPTY_MEAN = 0.589
PAPER_OPTIMAL_K = 3
PAPER_CRITICAL_VALUE = 0.5


def run_iris_experiment(verbose=True):
    """
    Run Iris dataset experiment with simplified output
    
    Returns:
        dict: Experiment results
    """
    # Load Iris data
    iris = load_iris()
    data = iris.data
    target = iris.target
    
    n_samples = data.shape[0]
    n_features = data.shape[1]
    n_classes = len(np.unique(target))
    
    # Setup: FOD = {setosa, virginica}, unknown = {versicolor}
    known_classes = [0, 2]  # setosa, virginica
    unknown_classes = [1]   # versicolor
    n_known = len(known_classes)
    
    # Data split (following paper Section 4.1)
    np.random.seed(108)
    
    train_indices = []
    test_indices = []
    
    for cls in range(3):
        cls_indices = np.where(target == cls)[0]
        np.random.shuffle(cls_indices)
        
        if cls in known_classes:
            train_indices.extend(cls_indices[:40])
            test_indices.extend(cls_indices[40:])
        else:
            test_indices.extend(cls_indices[:30])
    
    train_data = data[train_indices]
    train_labels = target[train_indices]
    test_data = data[test_indices]
    test_labels = target[test_indices]
    
    # Run Fuzzy Gap Statistic (suppress internal output)
    fgs = FuzzyGapStatistic(critical_value=PAPER_CRITICAL_VALUE, max_iterations=100, random_seed=108)
    
    # Build TFN and generate GBPA
    fgs.gbpa_generator.build_tfn_models(train_data, train_labels)
    gbpa_list, m_empty_mean, statistics = fgs.generate_gbpa_and_analyze(test_data)
    
    fod_complete = fgs.is_fod_complete(m_empty_mean)
    
    optimal_k = n_known
    n_unknown = 0
    
    if not fod_complete:
        # Perform FGS clustering
        sampled_data = fgs.perform_monte_carlo_sampling(test_data, n_samples=20)
        optimal_k, fgs_results = fgs.determine_optimal_clusters(test_data, sampled_data, max_clusters=6)
        n_unknown = fgs.reconstruct_fod(n_known, optimal_k)
    
    total_targets = n_known + n_unknown
    actual_classes = len(np.unique(test_labels))
    correct = total_targets == actual_classes
    
    # Simplified output
    if verbose:
        print("=" * 80)
        print(f"Dataset: Iris | Samples: {n_samples} | Features: {n_features} | Classes: {n_classes}")
        print("=" * 80)
        print(f"Known classes: {known_classes} | Unknown classes: {unknown_classes}")
        print()
        print("--- Step 1: GBPA Analysis ---")
        print(f"m̄(∅) = {m_empty_mean:.4f} | Critical value p = {PAPER_CRITICAL_VALUE} | FOD Complete: {fod_complete}")
        print()
        print("--- Step 2: FGS Clustering ---")
        print(f"Optimal k = {optimal_k} | Known targets: {n_known} | Unknown targets: {n_unknown}")
        print()
        print("--- Verification ---")
        result_str = "✓ CORRECT" if correct else "✗ INCORRECT"
        print(f"Actual classes: {actual_classes} | Predicted classes: {total_targets} | Result: {result_str}")
        print()
        print("--- Paper Comparison ---")
        m_diff = abs(m_empty_mean - PAPER_M_EMPTY_MEAN)
        k_diff = abs(optimal_k - PAPER_OPTIMAL_K)
        m_ok = "✓" if m_diff < 0.01 else "✗"
        k_ok = "✓" if k_diff == 0 else "✗"
        print(f"m̄(∅): {m_empty_mean:.4f} vs {PAPER_M_EMPTY_MEAN} (diff: {m_diff:.4f}) {m_ok}")
        print(f"Optimal k: {optimal_k} vs {PAPER_OPTIMAL_K} (diff: {k_diff}) {k_ok}")
        print("=" * 80)
    
    return {
        'dataset': 'Iris',
        'n_samples': n_samples,
        'n_features': n_features,
        'n_classes': n_classes,
        'm_empty_mean': m_empty_mean,
        'fod_complete': fod_complete,
        'optimal_k': optimal_k,
        'n_known': n_known,
        'n_unknown': n_unknown,
        'total_targets': total_targets,
        'actual_classes': actual_classes,
        'correct': correct
    }


if __name__ == "__main__":
    results = run_iris_experiment()