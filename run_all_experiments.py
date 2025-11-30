"""
Run All Experiments - Reproduce Paper Table 8-9

This script reproduces experiments from the paper:
"Updating incomplete framework of target recognition database based on fuzzy gap statistic"

Tests on 7 UCI datasets (Paper Table 7):
1. Iris (150 samples, 4 attributes, 3 classes)
2. Glass (214 samples, 9 attributes, 6 classes)
3. Haberman (306 samples, 3 attributes, 2 classes)
4. Knowledge (403 samples, 5 attributes, 4 classes)
5. Robot (5456 samples, 24 attributes, 5 classes) - using subset
6. Seeds (210 samples, 7 attributes, 3 classes)
7. WDBC (569 samples, 30 attributes, 2 classes)
"""

import warnings
warnings.filterwarnings('ignore')

from typing import Dict, List


def run_all_experiments(verbose: bool = False) -> List[Dict]:
    """
    Run all 7 UCI dataset experiments
    
    Args:
        verbose: If True, print detailed output for each experiment
        
    Returns:
        List of experiment results
    """
    all_results = []
    
    print("=" * 80)
    print("Running All UCI Dataset Experiments (Paper Table 7-9)")
    print("=" * 80)
    print()
    
    # Import and run each experiment
    from example_iris import run_iris_experiment
    from example_glass import run_glass_experiment
    from example_haberman import run_haberman_experiment
    from example_knowledge import run_knowledge_experiment
    from example_robot import run_robot_experiment
    from example_seeds import run_seeds_experiment
    from example_wdbc import run_wdbc_experiment
    
    experiments = [
        ("Iris", run_iris_experiment),
        ("Glass", run_glass_experiment),
        ("Haberman", run_haberman_experiment),
        ("Knowledge", run_knowledge_experiment),
        ("Robot", run_robot_experiment),
        ("Seeds", run_seeds_experiment),
        ("WDBC", run_wdbc_experiment),
    ]
    
    for name, run_func in experiments:
        print(f"Running {name} experiment...")
        result = run_func(verbose=verbose)
        all_results.append(result)
        print(f"  Done. FOD Complete: {result['fod_complete']}, Optimal k: {result['optimal_k']}, Correct: {'✓' if result['correct'] else '✗'}")
        print()
    
    return all_results


def print_summary_tables(results: List[Dict]):
    """
    Print summary tables similar to Paper Table 8 and Table 9
    """
    print()
    print("=" * 80)
    print("SUMMARY: Table 8 - Optimal k Values by FGS Method")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Samples':<8} {'Features':<10} {'Classes':<8} {'m̄(∅)':<10} {'FGS k':<8}")
    print("-" * 80)
    
    for r in results:
        print(f"{r['dataset']:<12} {r['n_samples']:<8} {r['n_features']:<10} {r['n_classes']:<8} {r['m_empty_mean']:<10.4f} {r['optimal_k']:<8}")
    
    print("-" * 80)
    print()
    
    print("=" * 80)
    print("SUMMARY: Table 9 - Unknown Target Detection Results")
    print("=" * 80)
    print(f"{'Dataset':<12} {'Real k':<8} {'Known':<8} {'Unknown':<10} {'Predicted':<10} {'Correct':<10}")
    print("-" * 80)
    
    n_correct = 0
    for r in results:
        correct_str = '✓ Yes' if r['correct'] else '✗ No'
        if r['correct']:
            n_correct += 1
        print(f"{r['dataset']:<12} {r['actual_classes']:<8} {r['n_known']:<8} {r['n_unknown']:<10} {r['total_targets']:<10} {correct_str:<10}")
    
    print("-" * 80)
    print(f"Overall Accuracy: {n_correct}/{len(results)} ({100*n_correct/len(results):.1f}%)")
    print("=" * 80)
    print()
    
    # FOD Completeness Summary
    print("=" * 80)
    print("SUMMARY: FOD Completeness Detection")
    print("=" * 80)
    print(f"{'Dataset':<12} {'m̄(∅)':<10} {'p=0.5':<10} {'FOD Complete':<15} {'Action':<20}")
    print("-" * 80)
    
    for r in results:
        fod_str = 'Yes' if r['fod_complete'] else 'No'
        action = 'No FGS needed' if r['fod_complete'] else 'FGS clustering'
        compare = '≤' if r['fod_complete'] else '>'
        print(f"{r['dataset']:<12} {r['m_empty_mean']:<10.4f} {compare:<10} {fod_str:<15} {action:<20}")
    
    print("-" * 80)
    print("=" * 80)


if __name__ == "__main__":
    # Run all experiments with concise output
    results = run_all_experiments(verbose=False)
    
    # Print summary tables
    print_summary_tables(results)
