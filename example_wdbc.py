"""
WDBC dataset experiment - Simplified output format
基于论文 Table 7: WDBC (569 samples, 30 attributes, 2 classes)

WDBC = Wisconsin Diagnostic Breast Cancer dataset
"""

import numpy as np
from fuzzy_gap_statistic import FuzzyGapStatistic

PAPER_CRITICAL_VALUE = 0.5


def load_wdbc_dataset():
    """Load Wisconsin Diagnostic Breast Cancer dataset"""
    try:
        from sklearn.datasets import load_breast_cancer
        dataset = load_breast_cancer()
        X = dataset.data
        y = dataset.target
        class_names = list(dataset.target_names)  # ['malignant', 'benign']
    except Exception:
        # Fallback: create synthetic WDBC-like data
        np.random.seed(108)
        n_malignant = 212
        n_benign = 357
        n_features = 30
        X = np.vstack([
            np.random.randn(n_malignant, n_features) * 2 + 1,
            np.random.randn(n_benign, n_features) * 1.5 + 0.5
        ])
        y = np.array([0] * n_malignant + [1] * n_benign)
        class_names = ['malignant', 'benign']
    
    return X, y, class_names


def run_wdbc_experiment(verbose=True):
    """
    Run WDBC (Wisconsin Diagnostic Breast Cancer) dataset experiment
    
    Paper Table 7: WDBC - 569 samples, 30 attributes, 2 classes
    """
    X, y, class_names = load_wdbc_dataset()
    
    n_samples = X.shape[0]
    n_features = X.shape[1]
    all_classes = np.unique(y)
    n_classes = len(all_classes)
    
    # Setup: Hide one class (malignant = 0)
    hidden_classes = [0]  # Hide malignant class
    known_classes = [c for c in all_classes if c not in hidden_classes]
    n_known = len(known_classes)
    
    # Data split
    np.random.seed(108)
    train_indices = []
    test_indices = []
    
    for cls in all_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        
        if cls in known_classes:
            n_train = int(len(cls_indices) * 0.8)
            train_indices.extend(cls_indices[:n_train])
            test_indices.extend(cls_indices[n_train:])
        else:
            n_test = min(30, len(cls_indices))
            test_indices.extend(cls_indices[:n_test])
    
    train_data = X[train_indices]
    train_labels = y[train_indices]
    test_data = X[test_indices]
    test_labels = y[test_indices]
    
    # Run FGS
    fgs = FuzzyGapStatistic(critical_value=PAPER_CRITICAL_VALUE, max_iterations=100, random_seed=108)
    
    fgs.gbpa_generator.build_tfn_models(train_data, train_labels)
    gbpa_list, m_empty_mean, statistics = fgs.generate_gbpa_and_analyze(test_data)
    
    fod_complete = fgs.is_fod_complete(m_empty_mean)
    
    optimal_k = n_known
    n_unknown = 0
    
    if not fod_complete:
        sampled_data = fgs.perform_monte_carlo_sampling(test_data, n_samples=20)
        optimal_k, fgs_results = fgs.determine_optimal_clusters(test_data, sampled_data, max_clusters=5)
        n_unknown = fgs.reconstruct_fod(n_known, optimal_k)
    
    total_targets = n_known + n_unknown
    actual_classes = len(np.unique(test_labels))
    correct = total_targets == actual_classes
    
    if verbose:
        print("=" * 80)
        print(f"Dataset: WDBC | Samples: {n_samples} | Features: {n_features} | Classes: {n_classes}")
        print("=" * 80)
        print(f"Known classes: {list(known_classes)} | Unknown classes: {hidden_classes}")
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
        print("=" * 80)
    
    return {
        'dataset': 'WDBC',
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
    results = run_wdbc_experiment()
