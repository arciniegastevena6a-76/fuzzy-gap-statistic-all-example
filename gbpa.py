"""
Generalized Basic Probability Assignment (GBPA) generation module
严格按照邓勇的强约束GBPA生成方法和GCR组合规则
关键修正：区分属性级m(Φ)平均值和GCR组合后的m(Φ)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List

# Numerical tolerance constants
GCR_COMPLETE_CONFLICT_TOLERANCE = 1e-10  # Tolerance for K=1 check in GCR (9-1 document formula 14)
MIN_GBPA_SUM = 1e-10  # Minimum sum to avoid division by zero
DEGENERATE_CLUSTER_THRESHOLD = 1e-6  # Threshold for detecting degenerate clusters


class GBPAGenerator:
    """
    GBPA Generator based on Triangular Fuzzy Numbers
    严格按照图8-11中的强约束方法和GCR组合规则
    """

    def __init__(self):
        """Initialize GBPA Generator"""
        self.tfn_models = {}  # {class_label: {feature_idx: (min, mean, max)}}
        self.known_classes = []

    def build_tfn_models(self, train_data: np.ndarray,
                        train_labels: np.ndarray) -> Dict:
        """
        Build Triangular Fuzzy Number models
        按照图3-4的方式构建TFN（对每个类别、每个属性）

        Args:
            train_data: Training data (n_samples, n_features) - 原始数据
            train_labels: Training labels (n_samples,)

        Returns:
            tfn_models: Dictionary of TFN models
        """
        self.known_classes = sorted(np.unique(train_labels))
        self.tfn_models = {}

        n_features = train_data.shape[1]

        for cls in self.known_classes:
            cls_mask = train_labels == cls
            cls_data = train_data[cls_mask]

            if len(cls_data) == 0:
                continue

            features_tfn = {}
            for j in range(n_features):
                feature_values = cls_data[:, j]
                min_val = np.min(feature_values)
                mean_val = np.mean(feature_values)
                max_val = np.max(feature_values)

                features_tfn[j] = (min_val, mean_val, max_val)

            self.tfn_models[cls] = features_tfn

        return self.tfn_models

    def _triangular_membership(self, x: float, min_val: float,
                              mean_val: float, max_val: float) -> float:
        """
        Calculate triangular membership function
        图1的三角隶属函数 Eq.(5)
        """
        if x < min_val:
            return 0.0
        elif min_val <= x <= mean_val:
            if mean_val == min_val:
                return 1.0
            return (x - min_val) / (mean_val - min_val)
        elif mean_val < x <= max_val:
            if max_val == mean_val:
                return 1.0
            return (max_val - x) / (max_val - mean_val)
        else:
            return 0.0

    def _generate_gbpa_for_single_attribute(self, attribute_value: float,
                                           feature_idx: int) -> Dict[str, float]:
        """
        Generate GBPA for a single attribute
        严格按照图8-10中的强约束规则（2.2节）

        核心规则：
        ① 当样本与某命题的三角模糊数表示模型相交时，
           相交点的纵坐标即为该样本支持命题的GBPA
        ② 当样本与多个命题的三角模糊数表示模型相交时，
           纵坐标高点为该样本支持单子集命题的GBPA，
           纵坐标低点为该样本支持多子集命题的GBPA
        ③ 如果累积之和超过1，归一化且m(Φ)=0；
           如果累积之和未超过1，生成m(Φ)
        ⑤ m(Φ)的生成准则：当样本与所有命题表示的三角形模糊数
           没有相交时，m(Φ)=1
        """
        if not self.tfn_models:
            raise ValueError("TFN models not built.")

        # Step 1: 计算与每个类别TFN的隶属度（交点纵坐标）
        memberships = {}
        for cls in self.known_classes:
            if feature_idx not in self.tfn_models[cls]:
                memberships[cls] = 0.0
                continue

            min_val, mean_val, max_val = self.tfn_models[cls][feature_idx]
            membership = self._triangular_membership(
                attribute_value, min_val, mean_val, max_val
            )
            memberships[cls] = membership

        # Step 2: 找出所有有交点的类别
        active_classes = [cls for cls, mem in memberships.items() if mem > 0]

        gbpa = {}

        # 规则⑤：与所有命题都不相交
        if len(active_classes) == 0:
            gbpa['empty'] = 1.0
            return gbpa

        # Step 3: 按照强约束规则生成GBPA
        # 按隶属度降序排列
        sorted_classes = sorted(active_classes,
                              key=lambda c: memberships[c],
                              reverse=True)

        # 规则①②③：生成单子集和多子集命题
        if len(active_classes) == 1:
            # Rule ①: Only one class intersection
            cls = active_classes[0]
            gbpa[frozenset([cls])] = memberships[cls]

        elif len(active_classes) >= 2:
            # Rules ②③: Multiple class intersections
            # According to 9-2 document Section 2.2 and Figure 5:
            # - Higher ordinate points: GBPA for each single-subset proposition
            # - Lower ordinate points: GBPA for pairwise multi-subset propositions
            # 
            # Step 1: Each class gets single-subset GBPA with its own membership
            for cls in sorted_classes:
                gbpa[frozenset([cls])] = memberships[cls]
            
            # Step 2: Generate pairwise multi-subset propositions (according to paper Figure 5)
            # For each pair of intersecting classes, generate a multi-subset proposition
            # using the minimum membership of the pair (lower ordinate point)
            from itertools import combinations
            for pair in combinations(sorted_classes, 2):
                cls_i, cls_j = pair
                # Use the minimum membership of the two classes
                pair_membership = min(memberships[cls_i], memberships[cls_j])
                gbpa[frozenset(pair)] = pair_membership
            
            # Step 3: If 3 or more classes intersect, also generate the full set multi-subset
            if len(active_classes) >= 3:
                # Full set multi-subset uses the minimum membership
                all_subset = frozenset(sorted_classes)
                min_membership = memberships[sorted_classes[-1]]  # Already sorted descending
                gbpa[all_subset] = min_membership

        # Step 4: 规则④ - 归一化或生成m(Φ)
        total_support = sum(gbpa.values())

        if total_support > 1.0:
            # 累积之和超过1，归一化
            for key in list(gbpa.keys()):
                gbpa[key] = gbpa[key] / total_support
            gbpa['empty'] = 0.0
        else:
            # 累积之和未超过1，剩余分配给空集
            gbpa['empty'] = 1.0 - total_support

        return gbpa

    def _generalized_combination_rule(self, m1: Dict, m2: Dict) -> Dict:
        """
        GCR - 广义组合规则
        严格按照文献 9-1 公式(11)-(14)

        关键公式：
        m(A) = (1-m(Φ)) × Σ m₁(B)m₂(C) / (1-K), B∩C=A  ... 公式(11)
        K = Σ m₁(B)m₂(C), B∩C=Φ                        ... 公式(12)
        m(Φ) = m₁(Φ) × m₂(Φ)                           ... 公式(13)
        m(Φ) = 1 当且仅当 K = 1                        ... 公式(14)

        关键修正：K 的计算必须包含空集参与的冲突！
        - Φ ∩ {a} = Φ → 冲突
        - {a} ∩ Φ = Φ → 冲突
        - Φ ∩ Φ = Φ → 冲突
        """
        combined = {}

        # 提取 m(Φ)
        m1_empty = m1.get('empty', 0.0)
        m2_empty = m2.get('empty', 0.0)

        # 计算 m(Φ) [公式(13)]
        m_combined_empty = m1_empty * m2_empty

        # ========================================
        # 关键修正：计算冲突系数 K [公式(12)]
        # K = Σ m₁(B)m₂(C), 其中 B∩C=Φ
        # 空集与任何集合的交集都为空！
        # ========================================
        K = 0.0

        for B_key in m1.keys():
            for C_key in m2.keys():
                # 判断 B∩C 是否为空
                intersection_is_empty = False

                if B_key == 'empty' or C_key == 'empty':
                    # 空集与任何集合的交集为空 → 冲突
                    intersection_is_empty = True
                elif isinstance(B_key, frozenset) and isinstance(C_key, frozenset):
                    # 两个非空集的交集
                    if len(B_key & C_key) == 0:
                        intersection_is_empty = True

                if intersection_is_empty:
                    K += m1[B_key] * m2[C_key]

        # 完全冲突 [公式(14)]
        # m(Φ) = 1 当且仅当 K = 1
        if K >= 1.0 - GCR_COMPLETE_CONFLICT_TOLERANCE:
            combined['empty'] = 1.0
            return combined

        # 组合非空集 [公式(11)]
        # m(A) = (1-m(Φ)) × Σ m₁(B)m₂(C) / (1-K)
        factor = (1 - m_combined_empty) / (1 - K)

        for B_key in m1.keys():
            if B_key == 'empty':
                continue
            for C_key in m2.keys():
                if C_key == 'empty':
                    continue

                intersection = B_key & C_key

                if len(intersection) > 0:
                    if intersection not in combined:
                        combined[intersection] = 0.0
                    combined[intersection] += factor * m1[B_key] * m2[C_key]

        # 空集 [公式(13)]
        combined['empty'] = m_combined_empty

        return combined

    def generate_gbpa_for_sample(self, sample: np.ndarray) -> Tuple[Dict, float, float, List[float]]:
        """
        为单个样本生成GBPA

        关键修正：
        - m_empty_combined: GCR组合后的空集mass（用于后续分类）
        - m_empty_mean_attribute: 属性级空集mass的平均值（用于判断FOD完整性）

        流程（图17）：
        1. 对每个属性生成独立的GBPA（鉴别生成GBPA）
        2. 使用GCR逐个组合（GCR合成）

        Returns:
            combined_gbpa: 组合后的GBPA
            m_empty_combined: GCR组合后的样本级空集mass
            m_empty_mean_attribute: 属性级空集mass的平均值（用于判断FOD！）
            attribute_m_empty: 每个属性的空集mass列表
        """
        n_features = len(sample)

        # Step 1: 为每个属性生成GBPA（证据内部）
        attribute_gbpas = []
        attribute_m_empty = []

        for j in range(n_features):
            attr_gbpa = self._generate_gbpa_for_single_attribute(sample[j], j)
            attribute_gbpas.append(attr_gbpa)
            attribute_m_empty.append(attr_gbpa.get('empty', 0.0))

        # Step 2: 使用GCR逐个组合（证据间组合）
        combined_gbpa = attribute_gbpas[0]

        for i in range(1, len(attribute_gbpas)):
            combined_gbpa = self._generalized_combination_rule(
                combined_gbpa, attribute_gbpas[i]
            )

        # GCR组合后的 m(Φ)
        m_empty_combined = combined_gbpa.get('empty', 0.0)

        # 属性级 m(Φ) 的平均值（论文中用这个判断FOD完整性！）
        m_empty_mean_attribute = np.mean(attribute_m_empty)

        return combined_gbpa, m_empty_combined, m_empty_mean_attribute, attribute_m_empty

    def generate(self, data: np.ndarray,
                labels: Optional[np.ndarray] = None) -> Tuple[List[Dict], np.ndarray, np.ndarray, List[List[float]]]:
        """
        Generate GBPA for all test samples

        修正：返回两种m(Φ)
        - m_empty_combined_array: GCR组合后的（用于后续分类）
        - m_empty_mean_attribute_array: 属性级平均（用于判断FOD完整性）

        Returns:
            gbpa_list: List of GBPA dictionaries for all samples
            m_empty_combined_array: GCR组合后的空集mass数组
            m_empty_mean_attribute_array: 属性级平均空集mass数组（用于判断FOD！）
            attribute_m_empty_list: List of attribute-level empty masses for each sample
        """
        if not self.tfn_models:
            raise ValueError("TFN models not built.")

        n_samples = data.shape[0]
        gbpa_list = []
        m_empty_combined_array = np.zeros(n_samples)
        m_empty_mean_attribute_array = np.zeros(n_samples)
        attribute_m_empty_list = []

        for i in range(n_samples):
            sample = data[i]
            gbpa, m_empty_combined, m_empty_mean_attr, attr_m_empty = \
                self.generate_gbpa_for_sample(sample)

            gbpa_list.append(gbpa)
            m_empty_combined_array[i] = m_empty_combined
            m_empty_mean_attribute_array[i] = m_empty_mean_attr
            attribute_m_empty_list.append(attr_m_empty)

        return gbpa_list, m_empty_combined_array, m_empty_mean_attribute_array, attribute_m_empty_list

    def calculate_mean_empty_mass(self, m_empty_array: np.ndarray) -> float:
        """
        计算平均空集mass（所有样本间）
        论文 Eq. (20)
        """
        return np.mean(m_empty_array)

    def calculate_mean_empty_mass_by_paper(self, data: np.ndarray) -> Tuple[float, List[float]]:
        """
        Calculate m̄(∅) exactly as shown in Paper Table 2
        
        According to Paper Eq. (20):
        m̄(∅) = (1/t) × Σᵢ₌₁ᵗ m(∅)
        where t is the number of attributes
        
        This method:
        1. For each attribute j, calculate the average m(∅) across all test samples
        2. Then average these attribute-level means to get the final m̄(∅)
        
        Args:
            data: Test data (n_samples, n_features)
            
        Returns:
            m_empty_mean: Final m̄(∅) value
            attribute_means: List of per-attribute m(∅) means
        """
        n_samples = data.shape[0]
        n_features = data.shape[1]
        
        # Store m(∅) for each attribute across all samples
        attribute_m_empty = {j: [] for j in range(n_features)}
        
        for i in range(n_samples):
            sample = data[i]
            for j in range(n_features):
                attr_gbpa = self._generate_gbpa_for_single_attribute(sample[j], j)
                m_empty = attr_gbpa.get('empty', 0.0)
                attribute_m_empty[j].append(m_empty)
        
        # Calculate mean for each attribute (like Table 2 rows)
        attribute_means = []
        for j in range(n_features):
            attr_mean = np.mean(attribute_m_empty[j])
            attribute_means.append(attr_mean)
        
        # Final m̄(∅) is mean of attribute means
        m_empty_mean = np.mean(attribute_means)
        
        return m_empty_mean, attribute_means

    def analyze_empty_mass_statistics(self, m_empty_combined_array: np.ndarray,
                                     m_empty_mean_attribute_array: np.ndarray,
                                     attribute_m_empty_list: List[List[float]]) -> Dict:
        """
        分析空集mass的统计特征
        用于判断系统状态（图17的判断逻辑）

        修正：区分GCR组合后的m(Φ)和属性级平均m(Φ)

        Returns:
            statistics: 包含各级别统计信息的字典
        """
        # 属性级平均的统计（用于判断FOD完整性）
        attribute_mean_level_mean = np.mean(m_empty_mean_attribute_array)
        attribute_mean_level_std = np.std(m_empty_mean_attribute_array)

        # GCR组合后的统计（用于参考）
        combined_level_mean = np.mean(m_empty_combined_array)
        combined_level_std = np.std(m_empty_combined_array)

        # 属性内统计（证据内部）
        attribute_m_empty_array = np.array(attribute_m_empty_list)
        attribute_level_mean = np.mean(attribute_m_empty_array, axis=0)
        attribute_level_overall_mean = np.mean(attribute_m_empty_array)

        statistics = {
            'attribute_mean_level': {  # 用于判断FOD完整性
                'mean': attribute_mean_level_mean,
                'std': attribute_mean_level_std,
                'min': np.min(m_empty_mean_attribute_array),
                'max': np.max(m_empty_mean_attribute_array)
            },
            'combined_level': {  # GCR组合后（仅供参考）
                'mean': combined_level_mean,
                'std': combined_level_std,
                'min': np.min(m_empty_combined_array),
                'max': np.max(m_empty_combined_array)
            },
            'attribute_level': {  # 每个属性的统计
                'per_attribute_mean': attribute_level_mean,
                'overall_mean': attribute_level_overall_mean,
                'max_mean': np.max(attribute_level_mean),
                'min_mean': np.min(attribute_level_mean)
            }
        }

        return statistics


def test_gcr_example4():
    """
    验证 GCR 计算 - 文献 9-1 例4
    
    输入:
        m1(a) = 0.1, m1(b) = 0.2, m1(Φ) = 0.7
        m2(a) = 0.1, m2({b,c}) = 0.1, m2(Φ) = 0.8
    
    正确的 K 计算（包含空集参与的冲突）:
        K = m1(a) * (m2({b,c}) + m2(Φ)) + m1(b) * (m2(a) + m2(Φ)) + m1(Φ) * (m2(a) + m2({b,c}) + m2(Φ))
        K = 0.1*0.9 + 0.2*0.9 + 0.7*1.0 = 0.97
    
    期望结果:
        m(Φ) = 0.7 * 0.8 = 0.56
        m(a) = (1-0.56) * 0.1*0.1 / (1-0.97) = 0.44 * 0.01 / 0.03 ≈ 0.147
        m(b) = (1-0.56) * 0.2*0.1 / (1-0.97) = 0.44 * 0.02 / 0.03 ≈ 0.293
    """
    m1 = {
        frozenset(['a']): 0.1,
        frozenset(['b']): 0.2,
        'empty': 0.7
    }
    m2 = {
        frozenset(['a']): 0.1,
        frozenset(['b', 'c']): 0.1,
        'empty': 0.8
    }
    
    generator = GBPAGenerator()
    result = generator._generalized_combination_rule(m1, m2)
    
    # 验证结果
    m_empty = result.get('empty', 0)
    m_a = result.get(frozenset(['a']), 0)
    m_b = result.get(frozenset(['b']), 0)
    
    assert abs(m_empty - 0.56) < 1e-3, f"m(Φ) 错误: {m_empty}, 期望 0.56"
    assert abs(m_a - 0.147) < 1e-3, f"m(a) 错误: {m_a}, 期望 0.147"
    assert abs(m_b - 0.293) < 1e-3, f"m(b) 错误: {m_b}, 期望 0.293"
    
    print("GCR 验证通过！结果符合文献 9-1 例4")
    return True


def test_pairwise_multisubset_proposition():
    """
    验证多子集命题生成 - 按照论文9-2 图5的成对多子集生成规则
    
    当样本与3个类(a, b, c)相交，隶属度分别为:
        memberships[a] = 0.8
        memberships[b] = 0.5
        memberships[c] = 0.3
    
    生成的GBPA应该包含:
        - 单子集命题: m({a}), m({b}), m({c})
        - 成对多子集命题: m({a,b}), m({b,c}), m({a,c})
        - 全集多子集命题: m({a,b,c})
    
    成对多子集使用两者中较小的隶属度:
        m({a,b}) = min(0.8, 0.5) = 0.5
        m({b,c}) = min(0.5, 0.3) = 0.3
        m({a,c}) = min(0.8, 0.3) = 0.3
        m({a,b,c}) = 0.3
    
    归一化前总和 = 0.8 + 0.5 + 0.3 + 0.5 + 0.3 + 0.3 + 0.3 = 3.0 > 1
    因此需要归一化，m(∅) = 0
    """
    generator = GBPAGenerator()
    
    # Set up TFN models that produce memberships of 0.8, 0.5, 0.3 at x=2
    # Using the formula: at x > mean, membership = (max-x)/(max-mean)
    # For class 0: (max-2)/(max-1) = 0.8 -> max = 6
    # For class 1: (max-2)/(max-1) = 0.5 -> max = 3
    # For class 2: (max-2)/(max-1) = 0.3 -> max ≈ 2.43
    generator.known_classes = [0, 1, 2]
    generator.tfn_models = {
        0: {0: (0.0, 1.0, 6.0)},    # At x=2, membership = 0.8
        1: {0: (0.0, 1.0, 3.0)},    # At x=2, membership = 0.5
        2: {0: (0.0, 1.0, 2.43)}    # At x=2, membership ≈ 0.3
    }
    
    # Generate GBPA at attribute value 2.0 for feature index 0
    gbpa = generator._generate_gbpa_for_single_attribute(2.0, 0)
    
    # Verify all focal elements are present
    expected_focal_elements = [
        frozenset([0]), frozenset([1]), frozenset([2]),           # Single subsets
        frozenset([0, 1]), frozenset([0, 2]), frozenset([1, 2]),  # Pairwise subsets
        frozenset([0, 1, 2]),                                     # Full set
        'empty'
    ]
    
    for fe in expected_focal_elements:
        assert fe in gbpa, f"焦元 {fe} 缺失!"
    
    # Verify m(empty) = 0 (due to normalization)
    assert abs(gbpa['empty'] - 0.0) < 1e-6, f"m(Φ) 应为 0，实际为 {gbpa['empty']}"
    
    # Verify total (excluding empty) sums to 1.0
    total = sum(v for k, v in gbpa.items() if k != 'empty')
    assert abs(total - 1.0) < 1e-6, f"总和应为 1.0，实际为 {total}"
    
    # Verify the structure follows the expected pattern
    # After normalization by 3.0:
    # m({0}) = 0.8/3.0 ≈ 0.267
    # m({1}) = 0.5/3.0 ≈ 0.167
    # m({2}) = 0.3/3.0 ≈ 0.100
    # m({0,1}) = 0.5/3.0 ≈ 0.167
    # m({0,2}) = 0.3/3.0 ≈ 0.100
    # m({1,2}) = 0.3/3.0 ≈ 0.100
    # m({0,1,2}) = 0.3/3.0 ≈ 0.100
    
    print("成对多子集命题生成验证通过！符合论文9-2 图5规则")
    return True


if __name__ == "__main__":
    # Run verification tests
    test_gcr_example4()
    test_pairwise_multisubset_proposition()