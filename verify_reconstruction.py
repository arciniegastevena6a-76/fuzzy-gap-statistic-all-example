"""
Step 3 FOD Reconstruction Verification Module

According to paper Section 3 Step 3:
"Judge again whether the FOD is complete. After determining the number of 
targets in the incomplete FOD, it needs to be reconstructed. By repeating 
step 1, the new triangular fuzzy number models can be established. Then 
generate the GBPAs again. Finally calculate the m̄(∅) again. The result is 
recorded in Table 6. Now the value of m̄(∅) is less than p = 1/2. Therefore, 
the FOD has been complete."

This module provides functions for:
1. Assigning pseudo-labels to unknown samples using FCM clustering
2. Rebuilding TFN models with combined known + pseudo-labeled data
3. Regenerating GBPA and verifying m̄(∅) < p
"""

import numpy as np
from typing import Dict, Tuple

from gbpa import GBPAGenerator
from fcm import FuzzyCMeans
from utils import standardize_data


def assign_pseudo_labels(test_data: np.ndarray,
                        optimal_k: int,
                        n_known_targets: int,
                        random_seed: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """
    Assign pseudo-labels to test samples using FCM clustering
    
    According to the paper, after determining the optimal number of clusters k,
    we use FCM to cluster the test data and assign pseudo-labels to samples
    that belong to unknown clusters.
    
    Args:
        test_data: Test dataset (n_samples, n_features)
        optimal_k: Optimal number of clusters from FGS
        n_known_targets: Number of known targets in the FOD
        random_seed: Random seed for reproducibility
        
    Returns:
        pseudo_labels: Array of pseudo-labels for each test sample
        cluster_centers: FCM cluster centers
    """
    # Standardize test data
    test_data_std = standardize_data(test_data)
    
    # Run FCM with optimal k
    fcm = FuzzyCMeans(n_clusters=optimal_k, max_iter=100, random_seed=random_seed)
    fcm.fit(test_data_std)
    
    # Get cluster assignments
    pseudo_labels = fcm.predict(test_data_std)
    
    return pseudo_labels, fcm.centers


def build_augmented_tfn_models(train_data: np.ndarray,
                               train_labels: np.ndarray,
                               test_data: np.ndarray,
                               pseudo_labels: np.ndarray,
                               n_known_targets: int) -> GBPAGenerator:
    """
    Build augmented TFN models with known + pseudo-labeled data
    
    This corresponds to paper Step 3: "By repeating step 1, the new 
    triangular fuzzy number models can be established."
    
    Args:
        train_data: Original training data (known classes)
        train_labels: Original training labels
        test_data: Test dataset
        pseudo_labels: Pseudo-labels from FCM clustering
        n_known_targets: Number of known targets
        
    Returns:
        gbpa_generator: New GBPA generator with augmented TFN models
    """
    # Create augmented training data
    augmented_data = list(train_data)
    augmented_labels = list(train_labels)
    
    # Get known class labels
    known_classes = np.unique(train_labels)
    max_known_label = max(known_classes) if len(known_classes) > 0 else -1
    
    # Add test data with pseudo-labels as new classes
    for i, sample in enumerate(test_data):
        cluster_id = pseudo_labels[i]
        # Map cluster ID to new label space
        # New labels start after the maximum known label
        new_label = max_known_label + 1 + cluster_id
        augmented_data.append(sample)
        augmented_labels.append(new_label)
    
    augmented_data = np.array(augmented_data)
    augmented_labels = np.array(augmented_labels)
    
    # Build new TFN models
    gbpa_generator = GBPAGenerator()
    gbpa_generator.build_tfn_models(augmented_data, augmented_labels)
    
    return gbpa_generator


def verify_fod_completeness(test_data: np.ndarray,
                           gbpa_generator: GBPAGenerator,
                           critical_value: float = 0.5) -> Dict:
    """
    Verify if the reconstructed FOD is complete
    
    This corresponds to paper Step 3: "Finally calculate the m̄(∅) again.
    Now the value of m̄(∅) is less than p = 1/2. Therefore, the FOD has 
    been complete."
    
    Args:
        test_data: Test dataset
        gbpa_generator: GBPA generator with reconstructed TFN models
        critical_value: Threshold for FOD completeness (default 0.5)
        
    Returns:
        verification_results: Dictionary containing verification info
    """
    # Generate GBPA for test data
    gbpa_list, m_empty_combined, m_empty_mean_attr, attr_m_empty = \
        gbpa_generator.generate(test_data)
    
    # Calculate new m̄(∅) - attribute-level average
    new_m_empty_mean = np.mean(m_empty_mean_attr)
    
    # Check completeness
    fod_complete = new_m_empty_mean <= critical_value
    
    # Analyze per-attribute statistics
    attr_m_empty_array = np.array(attr_m_empty)
    per_attr_means = np.mean(attr_m_empty_array, axis=0)
    
    results = {
        'new_m_empty_mean': new_m_empty_mean,
        'fod_complete': fod_complete,
        'critical_value': critical_value,
        'per_attribute_m_empty_means': per_attr_means,
        'gbpa_list': gbpa_list,
        'm_empty_combined': m_empty_combined,
        'm_empty_mean_attr': m_empty_mean_attr
    }
    
    return results


def full_reconstruction_verification(test_data: np.ndarray,
                                     train_data: np.ndarray,
                                     train_labels: np.ndarray,
                                     optimal_k: int,
                                     n_known_targets: int,
                                     critical_value: float = 0.5,
                                     random_seed: int = 42,
                                     verbose: bool = True) -> Dict:
    """
    Full Step 3 reconstruction and verification process
    
    This function implements the complete Step 3 workflow from the paper:
    1. Use FCM clustering results to assign pseudo-labels
    2. Rebuild TFN models with all classes
    3. Regenerate GBPA and verify m̄(∅) < p
    
    Args:
        test_data: Test dataset
        train_data: Training dataset (known classes)
        train_labels: Training labels
        optimal_k: Optimal number of clusters from Step 2
        n_known_targets: Number of known targets
        critical_value: Threshold for FOD completeness
        random_seed: Random seed for reproducibility
        verbose: Whether to print progress
        
    Returns:
        results: Dictionary with all verification results
    """
    results = {}
    
    if verbose:
        print("\n=== Step 3: FOD Reconstruction Verification ===")
    
    # Step 3.1: Assign pseudo-labels using FCM
    if verbose:
        print(f"\nStep 3.1: Assigning pseudo-labels using FCM (k={optimal_k})...")
    
    pseudo_labels, cluster_centers = assign_pseudo_labels(
        test_data, optimal_k, n_known_targets, random_seed
    )
    
    results['pseudo_labels'] = pseudo_labels
    results['cluster_centers'] = cluster_centers
    
    if verbose:
        unique_labels = np.unique(pseudo_labels)
        print(f"  Found {len(unique_labels)} clusters in test data")
        for label in unique_labels:
            count = np.sum(pseudo_labels == label)
            print(f"    Cluster {label}: {count} samples")
    
    # Step 3.2: Build augmented TFN models
    if verbose:
        print("\nStep 3.2: Building augmented TFN models...")
    
    augmented_generator = build_augmented_tfn_models(
        train_data, train_labels, test_data, pseudo_labels, n_known_targets
    )
    
    results['augmented_classes'] = augmented_generator.known_classes
    
    if verbose:
        print(f"  Total classes in augmented model: {len(augmented_generator.known_classes)}")
        for cls in augmented_generator.known_classes:
            print(f"    Class {cls}")
    
    # Step 3.3: Verify FOD completeness
    if verbose:
        print("\nStep 3.3: Verifying FOD completeness...")
    
    verification = verify_fod_completeness(
        test_data, augmented_generator, critical_value
    )
    
    results.update(verification)
    
    if verbose:
        print(f"\n  New m̄(∅) after reconstruction: {verification['new_m_empty_mean']:.4f}")
        print(f"  Critical value p: {critical_value}")
        print(f"  Per-attribute m(∅) means:")
        for i, mean in enumerate(verification['per_attribute_m_empty_means']):
            print(f"    Attribute {i}: {mean:.4f}")
        
        if verification['fod_complete']:
            print(f"\n  ✓ FOD is now COMPLETE (m̄(∅) = {verification['new_m_empty_mean']:.4f} ≤ {critical_value})")
        else:
            print(f"\n  ⚠ FOD is still INCOMPLETE (m̄(∅) = {verification['new_m_empty_mean']:.4f} > {critical_value})")
    
    # Calculate number of unknown targets
    n_unknown = optimal_k - n_known_targets if optimal_k > n_known_targets else 0
    results['n_unknown_targets'] = n_unknown
    results['total_targets'] = n_known_targets + n_unknown
    
    return results


if __name__ == "__main__":
    # Test the verification module
    from sklearn.datasets import load_iris
    
    print("=" * 70)
    print("Testing Step 3 FOD Reconstruction Verification")
    print("=" * 70)
    
    # Load Iris data
    iris = load_iris()
    data = iris.data
    target = iris.target
    
    # Setup: Known classes are setosa (0) and virginica (2)
    # Unknown class is versicolor (1)
    known_classes = [0, 2]
    np.random.seed(42)
    
    # Data split
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
    
    print(f"\nTraining set: {len(train_data)} samples")
    print(f"Test set: {len(test_data)} samples")
    
    # Simulate that we've determined optimal_k = 3
    optimal_k = 3
    n_known_targets = len(known_classes)
    
    # Run full verification
    results = full_reconstruction_verification(
        test_data=test_data,
        train_data=train_data,
        train_labels=train_labels,
        optimal_k=optimal_k,
        n_known_targets=n_known_targets,
        critical_value=0.5,
        random_seed=42,
        verbose=True
    )
    
    print("\n" + "=" * 70)
    print("Verification Complete!")
    print("=" * 70)
