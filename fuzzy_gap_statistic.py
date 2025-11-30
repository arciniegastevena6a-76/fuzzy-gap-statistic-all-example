"""
Main module for Fuzzy Gap Statistic algorithm
严格按照论文Section 3和图17的完整流程
关键修正：使用属性级m(Φ)平均值判断FOD完整性
"""

import numpy as np
from typing import Tuple, Dict, Optional
from gbpa import GBPAGenerator


class FuzzyGapStatistic:
    """
    Main class implementing Fuzzy Gap Statistic algorithm
    论文 Section 3: Proposed Method
    """

    def __init__(self, critical_value: float = 0.5, max_iterations: int = 100,
                 random_seed: int = 108):
        """
        Initialize Fuzzy Gap Statistic

        Args:
            critical_value: Threshold for incomplete FOD (论文 p = 0.5)
            max_iterations: Maximum iterations for FCM
            random_seed: Random seed for reproducibility
        """
        self.critical_value = critical_value
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        self.gbpa_generator = GBPAGenerator()
        self.fcm = None
        self.gap_calc = None

    def generate_gbpa_and_analyze(self, data: np.ndarray) -> Tuple[list, float, Dict]:
        """
        Step 1: Generate GBPA and analyze statistics
        论文 Step 1.1 - 1.5 + 图17的判断逻辑

        关键修正：使用属性级m(Φ)的平均值来判断FOD完整性

        Args:
            data: Test data (原始数据，未标准化)

        Returns:
            gbpa_list: List of GBPA dictionaries
            m_empty_mean: Mean of empty set mass (属性级平均，用于判断FOD)
            statistics: 详细统计信息（用于判断系统状态）
        """
        gbpa_list, m_empty_combined_array, m_empty_mean_attribute_array, attribute_m_empty_list = \
            self.gbpa_generator.generate(data)

        # 用于判断FOD完整性的m(Φ)：属性级平均
        m_empty_mean_for_fod = np.mean(m_empty_mean_attribute_array)

        # 分析统计特征（对应图16-17的分析）
        statistics = self.gbpa_generator.analyze_empty_mass_statistics(
            m_empty_combined_array,
            m_empty_mean_attribute_array,
            attribute_m_empty_list
        )

        return gbpa_list, m_empty_mean_for_fod, statistics

    def diagnose_system_state(self, m_empty_mean: float,
                             statistics: Dict) -> Dict:
        """
        诊断系统状态（图17的完整判断逻辑）

        判断流程：
        1. 样本间m(Φ)是否大于阈值？
           ├─ 是 → 辨识框架不完整
           └─ 否 → 检查样本内传感器报告
                  ├─ 较大 → 辨识框架完整但传感器受干扰
                  └─ 都小 → 辨识框架完整

        Returns:
            diagnosis: 诊断结果字典
        """
        diagnosis = {}

        # 属性级平均的m(Φ)
        sample_mean = statistics['attribute_mean_level']['mean']
        attr_overall_mean = statistics['attribute_level']['overall_mean']
        attr_max_mean = statistics['attribute_level']['max_mean']

        # 判断1：样本间m(Φ)是否大于阈值（图17第一个菱形）
        if sample_mean > self.critical_value:
            diagnosis['state'] = 'incomplete_fod'
            diagnosis['description'] = '辨识框架不完整'
            diagnosis['confidence'] = 'high'
            diagnosis['reason'] = f'属性级平均m(Φ) ({sample_mean:.4f}) > 阈值 ({self.critical_value})'

        else:
            # 判断2：样本内传感器报告是否比较小（图17第二个菱形）
            if attr_max_mean > self.critical_value:
                # 存在某些属性的m(Φ)较大 → 可能是传感器受干扰
                diagnosis['state'] = 'sensor_jammed'
                diagnosis['description'] = '辨识框架完整但传感器受干扰'
                diagnosis['confidence'] = 'medium'
                diagnosis['reason'] = f'属性级平均m(Φ) ({sample_mean:.4f}) ≤ 阈值，但某些属性m(Φ) ({attr_max_mean:.4f}) 较大'
            else:
                # 所有指标都小 → 框架完整
                diagnosis['state'] = 'complete_fod'
                diagnosis['description'] = '辨识框架完整'
                diagnosis['confidence'] = 'high'
                diagnosis['reason'] = f'属性级平均m(Φ) ({sample_mean:.4f}) ≤ 阈值，属性级m(Φ) ({attr_max_mean:.4f}) 也较小'

        diagnosis['statistics'] = statistics

        return diagnosis

    def is_fod_complete(self, m_empty_mean: float) -> bool:
        """
        Check if FOD is complete
        论文 Step 1.5: 判断 m̄(∅) ≤ p
        """
        return m_empty_mean <= self.critical_value

    def perform_monte_carlo_sampling(self, data: np.ndarray,
                                     n_samples: int = 20) -> list:
        """
        Step 2.1: Perform Monte Carlo sampling
        论文：通常采样 20 次 (B=20 in paper)
        
        Args:
            data: Original test data
            n_samples: Number of Monte Carlo samples (default 20)
            
        Returns:
            sampled_data_list: List of sampled datasets
        """
        # 需要导入utils模块
        from utils import standardize_data
        data_std = standardize_data(data)

        from monte_carlo import MonteCarloSampling
        mc_sampler = MonteCarloSampling(random_seed=self.random_seed)
        sampled_data_list = []

        for i in range(n_samples):
            # Use different seed for each sample but reproducible
            if self.random_seed is not None:
                mc_sampler.set_seed(self.random_seed + i)
            sampled = mc_sampler.sample_uniform(data_std)
            sampled_data_list.append(sampled)

        return sampled_data_list

    def determine_optimal_clusters(self, original_data: np.ndarray,
                                   sampled_data_list: list,
                                   max_clusters: int = 10) -> Tuple[int, Dict]:
        """
        Step 2.2 - 2.3: Apply FCM and FGS
        论文 Algorithm 2: FGS
        
        Args:
            original_data: Original test data
            sampled_data_list: List of Monte Carlo sampled data
            max_clusters: Maximum number of clusters to test
            
        Returns:
            optimal_k: Optimal number of clusters
            fgs_results: Dictionary of FGS values for each k
        """
        from utils import standardize_data
        from fcm import FuzzyCMeans
        from gap_statistic import GapStatisticCalculator

        original_data_std = standardize_data(original_data)

        self.gap_calc = GapStatisticCalculator(
            max_iterations=self.max_iterations
        )

        fgs_results = {}

        for k in range(1, max_clusters + 1):
            log_jmk_star_list = []

            # Run FCM on each Monte Carlo sample
            for idx, sampled_data in enumerate(sampled_data_list):
                # Use different seed for each sample
                seed = self.random_seed + k * 100 + idx if self.random_seed else None
                fcm_star = FuzzyCMeans(n_clusters=k, max_iter=self.max_iterations,
                                       random_seed=seed)
                fcm_star.fit(sampled_data)

                obj_value = max(fcm_star.objective_value, 1e-10)
                log_jmk_star = np.log(obj_value)
                log_jmk_star_list.append(log_jmk_star)

            # Run FCM on original data (standardized)
            seed = self.random_seed + k * 1000 if self.random_seed else None
            fcm_original = FuzzyCMeans(n_clusters=k, max_iter=self.max_iterations,
                                       random_seed=seed)
            fcm_original.fit(original_data_std)
            self.fcm = fcm_original  # Store for later use
            obj_value = max(fcm_original.objective_value, 1e-10)
            log_jmk = np.log(obj_value)

            # Calculate FGS
            gap_k = self.gap_calc.calculate_gap(
                log_jmk_star_list, log_jmk, k
            )
            fgs_results[k] = gap_k

        # Determine optimal k
        optimal_k = self.gap_calc.find_optimal_k(fgs_results)

        return optimal_k, fgs_results

    def reconstruct_fod(self, n_known_targets: int,
                        optimal_k: int) -> int:
        """
        Step 2.4: Reconstruct FOD
        论文: 未知目标数 = optimal_k - 已知目标数
        """
        if optimal_k > n_known_targets:
            n_unknown_targets = optimal_k - n_known_targets
        else:
            n_unknown_targets = 0

        return n_unknown_targets

    def fit(self, test_data: np.ndarray,
            train_data: np.ndarray,
            train_labels: np.ndarray,
            n_known_targets: Optional[int] = None,
            max_clusters: int = 10) -> Dict:
        """
        Complete pipeline: Fit Fuzzy Gap Statistic
        严格按照图17的完整流程

        Args:
            test_data: Test dataset (原始数据，未标准化)
            train_data: Training dataset (原始数据，只包含已知类别)
            train_labels: Training labels
            n_known_targets: Number of known targets
            max_clusters: Maximum clusters to test

        Returns:
            results: Dictionary containing all results
        """
        results = {}

        # ===== Step 1: 构建TFN并判断FOD完整性 =====
        print("=" * 70)
        print("Step 1: Building TFN models and generating GBPA")
        print("=" * 70)

        # Step 1.1: 用训练数据构建 TFN 模型（建立命题表示模型）
        print("\nStep 1.1: Building TFN models from training data...")
        self.gbpa_generator.build_tfn_models(train_data, train_labels)

        if n_known_targets is None:
            n_known_targets = len(np.unique(train_labels))

        print(f"  Known targets: {n_known_targets}")
        print(f"  Known classes: {self.gbpa_generator.known_classes}")

        # 打印TFN模型信息（对应论文Table 3）
        print("\nTFN Models (corresponding to Table 3 in paper):")
        for cls in self.gbpa_generator.known_classes:
            print(f"\n  Class {cls}:")
            for feat_idx in range(len(self.gbpa_generator.tfn_models[cls])):
                min_v, mean_v, max_v = self.gbpa_generator.tfn_models[cls][feat_idx]
                print(f"    Feature {feat_idx}: ({min_v:.3f}, {mean_v:.3f}, {max_v:.3f})")

        # Step 1.2-1.4: 用测试数据生成 GBPA 并分析统计特征
        print("\nStep 1.2-1.4: Generating GBPA and analyzing statistics...")
        gbpa_list, m_empty_mean, statistics = self.generate_gbpa_and_analyze(test_data)

        results['gbpa_list'] = gbpa_list
        results['m_empty_mean'] = m_empty_mean
        results['statistics'] = statistics

        # 打印详细统计信息（对应图16的分析）
        print("\n=== GBPA Generation Statistics (Fig.16 analysis) ===")
        print(f"\n① 证据内部（Within evidence - each attribute）:")
        for i, mean in enumerate(statistics['attribute_level']['per_attribute_mean']):
            print(f"   Attribute {i}: m(Φ) mean = {mean:.4f}")
        print(f"   Overall attribute-level mean: {statistics['attribute_level']['overall_mean']:.4f}")

        print(f"\n② 一个样本内证据间（Among evidences of a sample - 属性级平均）:")
        print(f"   Mean: {statistics['attribute_mean_level']['mean']:.4f}  (用于判断FOD完整性)")
        print(f"   Std:  {statistics['attribute_mean_level']['std']:.4f}")
        print(f"   Min:  {statistics['attribute_mean_level']['min']:.4f}")
        print(f"   Max:  {statistics['attribute_mean_level']['max']:.4f}")

        print(f"\n   GCR组合后（仅供参考）:")
        print(f"   Mean: {statistics['combined_level']['mean']:.4f}")
        print(f"   Std:  {statistics['combined_level']['std']:.4f}")

        # Step 1.5: 诊断系统状态（图17的完整判断逻辑）
        print("\nStep 1.5: Diagnosing system state (Fig.17 logic)...")
        diagnosis = self.diagnose_system_state(m_empty_mean, statistics)

        results['diagnosis'] = diagnosis
        results['fod_is_complete'] = self.is_fod_complete(m_empty_mean)

        print(f"\n=== Diagnosis Result ===")
        print(f"  State: {diagnosis['state']}")
        print(f"  Description: {diagnosis['description']}")
        print(f"  Confidence: {diagnosis['confidence']}")
        print(f"  Reason: {diagnosis['reason']}")
        print(f"  m̄(∅) = {m_empty_mean:.4f}  (论文中应为 0.589)")
        print(f"  Critical value p = {self.critical_value}")
        print(f"  FOD Complete: {results['fod_is_complete']}")

        if results['fod_is_complete']:
            print("\n✓ FOD is complete. No need to reconstruct.")
            results['n_unknown_targets'] = 0
            results['optimal_k'] = n_known_targets
            return results

        # ===== Step 2: Monte Carlo采样和FGS =====
        print("\n" + "=" * 70)
        print("Step 2: FOD is incomplete, applying FGS")
        print("=" * 70)

        # Step 2.1: Monte Carlo采样
        print("\nStep 2.1: Performing Monte Carlo sampling...")
        sampled_data = self.perform_monte_carlo_sampling(test_data, n_samples=20)
        print(f"  Generated {len(sampled_data)} Monte Carlo samples")

        # Step 2.2-2.3: FCM聚类和FGS计算
        print("\nStep 2.2-2.3: Running FCM and calculating FGS...")
        optimal_k, fgs_results = self.determine_optimal_clusters(
            test_data, sampled_data, max_clusters
        )

        results['fgs_results'] = fgs_results
        results['optimal_k'] = optimal_k

        print("\nFuzzy Gap Statistic Results (corresponding to Fig.4):")
        print(f"{'k':<5} {'Gap(k)':<12} {'s_k':<12}")
        print("-" * 30)
        for k in sorted(fgs_results.keys()):
            gap = fgs_results[k]
            s_k = self.gap_calc.s_k.get(k, 0.0)
            print(f"{k:<5} {gap:<12.6f} {s_k:<12.6f}")

        print(f"\n✓ Optimal k = {optimal_k} (Fig.4 in paper: k=3)")

        # Step 2.4: 重构FOD
        n_unknown = self.reconstruct_fod(n_known_targets, optimal_k)
        results['n_unknown_targets'] = n_unknown
        results['total_targets'] = n_known_targets + n_unknown

        print(f"\nStep 2.4: Reconstructing FOD")
        print(f"  Known targets: {n_known_targets}")
        print(f"  Unknown targets: {n_unknown}")
        print(f"  Total targets: {results['total_targets']}")

        # ===== Step 3: 重新构建FOD并验证 =====
        if n_unknown > 0:
            print("\n" + "=" * 70)
            print("Step 3: Reconstructing complete FOD and verifying")
            print("=" * 70)

            # Step 3.1: Assign pseudo-labels to unknown samples using FCM
            step3_results = self.verify_reconstruction(
                test_data=test_data,
                train_data=train_data,
                train_labels=train_labels,
                optimal_k=optimal_k,
                n_known_targets=n_known_targets
            )
            
            results['step3_verification'] = step3_results
            
            print(f"\n=== Step 3 Verification Results ===")
            print(f"  New m̄(∅) after reconstruction: {step3_results['new_m_empty_mean']:.4f}")
            print(f"  Critical value p: {self.critical_value}")
            print(f"  FOD now complete: {step3_results['fod_now_complete']}")
            
            if step3_results['fod_now_complete']:
                print(f"\n✓ FOD reconstruction verified! m̄(∅) < p")
            else:
                print(f"\n⚠ FOD may still be incomplete. Consider more iterations.")

        return results

    def verify_reconstruction(self, test_data: np.ndarray,
                             train_data: np.ndarray,
                             train_labels: np.ndarray,
                             optimal_k: int,
                             n_known_targets: int) -> Dict:
        """
        Step 3: Verify FOD reconstruction
        论文 Step 3: Judge again whether the FOD is complete
        
        After determining the number of targets in the incomplete FOD,
        it needs to be reconstructed. By repeating step 1, the new 
        triangular fuzzy number models can be established. Then generate 
        the GBPAs again. Finally calculate the m̄(∅) again.
        
        Args:
            test_data: Test dataset
            train_data: Training dataset (known classes)
            train_labels: Training labels
            optimal_k: Optimal number of clusters from FGS
            n_known_targets: Number of known targets
            
        Returns:
            verification_results: Dictionary with verification info
        """
        from utils import standardize_data
        from fcm import FuzzyCMeans
        
        results = {}
        
        # Step 3.1: Use FCM to cluster test data with optimal k
        test_data_std = standardize_data(test_data)
        fcm = FuzzyCMeans(n_clusters=optimal_k, max_iter=self.max_iterations,
                         random_seed=self.random_seed)
        fcm.fit(test_data_std)
        
        # Get cluster assignments (pseudo-labels)
        pseudo_labels = fcm.predict(test_data_std)
        
        # Step 3.2: Create augmented training data
        # Combine known data with pseudo-labeled unknown data
        # Map pseudo-labels to new class indices starting after known classes
        known_classes = np.unique(train_labels)
        n_known = len(known_classes)
        
        # Create mapping: assign new class indices to clusters
        # that don't correspond to known classes
        augmented_train_data = list(train_data)
        augmented_train_labels = list(train_labels)
        
        # Add test data with pseudo-labels as new classes
        for i, sample in enumerate(test_data):
            cluster_id = pseudo_labels[i]
            # Assign pseudo-label: known classes + cluster_id
            # This creates new class labels for unknown clusters
            new_label = n_known + cluster_id
            augmented_train_data.append(sample)
            augmented_train_labels.append(new_label)
        
        augmented_train_data = np.array(augmented_train_data)
        augmented_train_labels = np.array(augmented_train_labels)
        
        # Step 3.3: Rebuild TFN models with all classes
        new_gbpa_generator = GBPAGenerator()
        new_gbpa_generator.build_tfn_models(augmented_train_data, augmented_train_labels)
        
        # Step 3.4: Regenerate GBPA for test data
        gbpa_list, m_empty_combined_array, m_empty_mean_attribute_array, _ = \
            new_gbpa_generator.generate(test_data)
        
        # Step 3.5: Calculate new m̄(∅)
        new_m_empty_mean = np.mean(m_empty_mean_attribute_array)
        
        results['new_m_empty_mean'] = new_m_empty_mean
        results['fod_now_complete'] = new_m_empty_mean <= self.critical_value
        results['pseudo_labels'] = pseudo_labels
        results['augmented_classes'] = np.unique(augmented_train_labels)
        results['gbpa_list'] = gbpa_list
        
        return results