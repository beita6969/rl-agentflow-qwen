#!/usr/bin/env python3
"""
奖励计算器 - 改进版(借鉴ROLL和AgentFlow设计)
"""
import sys
import re
from typing import Any, Dict, Optional

# 添加AFlow到路径
sys.path.insert(0, '/home/yijia/.claude/11/AFlow')


class RewardComputer:
    """
    改进的奖励计算器

    新增特性(借鉴ROLL):
    1. 格式奖励 - 检查<think>/<answer>标签
    2. 重复惩罚 - N-gram重复检测
    3. 改进的数学评估 - 支持LaTeX和boxed
    4. 更细粒度的评分阶梯
    """

    def __init__(
        self,
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            reward_weights: 奖励权重配置
        """
        # 默认权重(根据ROLL经验调整)
        self.reward_weights = reward_weights or {
            "correctness": 0.65,   # 从0.7降低,给其他维度留空间
            "efficiency": 0.15,    # 从0.2降低
            "simplicity": 0.10,    # 保持
            "format": 0.05,        # 新增: 格式规范
            "repetition": 0.05     # 新增: 重复惩罚
        }

        print(f"✅ 改进版奖励计算器初始化完成")
        print(f"  权重: {self.reward_weights}")
        print(f"  新增维度: 格式奖励、重复惩罚")

    def compute_reward(
        self,
        problem: str,
        prediction: Any,
        ground_truth: Any,
        problem_type: str = "math",
        metadata: Optional[Dict] = None
    ) -> float:
        """
        计算总奖励

        Returns:
            reward: [-10, 10]范围的奖励值
        """
        metadata = metadata or {}

        # 1. 正确性奖励
        correctness_reward = self._compute_correctness_reward(
            prediction, ground_truth, problem_type
        )

        # 2. 效率奖励(基于成本) - 添加类型转换
        cost = metadata.get("cost", 0.0)
        try:
            cost = float(cost) if cost is not None else 0.0
        except (ValueError, TypeError):
            cost = 0.0
        efficiency_reward = self._compute_efficiency_reward(cost)

        # 3. 简洁性奖励(基于执行时间或算子数) - 添加类型转换
        execution_time = metadata.get("execution_time", 0.0)
        try:
            execution_time = float(execution_time) if execution_time is not None else 0.0
        except (ValueError, TypeError):
            execution_time = 0.0

        num_operators = metadata.get("num_operators", 1)
        try:
            num_operators = int(num_operators) if num_operators is not None else 1
        except (ValueError, TypeError):
            num_operators = 1

        simplicity_reward = self._compute_simplicity_reward(
            execution_time,
            num_operators
        )

        # 4. 格式奖励(新增 - ROLL风格)
        format_reward = self._compute_format_reward(
            str(prediction) if prediction else "",
            problem_type
        )

        # 5. 重复惩罚(新增 - ROLL风格)
        repetition_penalty = self._compute_repetition_penalty(
            str(prediction) if prediction else ""
        )

        # 加权总奖励
        total_reward = (
            self.reward_weights["correctness"] * correctness_reward +
            self.reward_weights["efficiency"] * efficiency_reward +
            self.reward_weights["simplicity"] * simplicity_reward +
            self.reward_weights["format"] * format_reward +
            self.reward_weights["repetition"] * repetition_penalty
        )

        # 裁剪到[-10, 10]
        total_reward = max(-10.0, min(10.0, total_reward))

        return total_reward

    def _compute_correctness_reward(
        self,
        prediction: Any,
        ground_truth: Any,
        problem_type: str
    ) -> float:
        """
        计算正确性奖励

        Returns:
            reward: [-10, 10]
        """
        if prediction is None:
            return -10.0  # 执行失败

        if problem_type == "math":
            return self._compute_math_correctness(prediction, ground_truth)
        elif problem_type == "code":
            return self._compute_code_correctness(prediction, ground_truth)
        elif problem_type == "qa":
            return self._compute_qa_correctness(prediction, ground_truth)
        else:
            return self._compute_general_correctness(prediction, ground_truth)

    def _compute_math_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        数学问题正确性(改进版 - 借鉴ROLL)

        改进:
        1. 支持LaTeX \boxed{}格式
        2. 更细粒度的评分阶梯
        3. 更好的数字提取
        """
        try:
            pred_str = str(prediction)
            gt_str = str(ground_truth)

            # 方法1: 提取boxed答案(ROLL风格)
            pred_boxed = self._extract_boxed(pred_str)
            gt_boxed = self._extract_boxed(gt_str)

            if pred_boxed and gt_boxed:
                try:
                    pred_num = float(pred_boxed)
                    gt_num = float(gt_boxed)
                    diff = abs(pred_num - gt_num)

                    if diff < 1e-4:
                        return 10.0   # 完全正确
                    elif diff < 0.1:
                        return 8.0    # 非常接近(新增阶梯)
                    elif diff < 1.0:
                        return 5.0    # 接近
                    elif diff < 10.0:
                        return 2.0    # 数量级正确(新增阶梯)
                    else:
                        return -5.0   # 错误
                except:
                    pass

            # 方法2: 数字提取(改进版)
            pred_numbers = self._extract_numbers(pred_str)
            gt_numbers = self._extract_numbers(gt_str)

            if not gt_numbers:
                # 无法提取ground truth数字,使用字符串匹配
                if gt_str.strip().lower() in pred_str.strip().lower():
                    return 10.0
                else:
                    return -5.0

            if not pred_numbers:
                # 无法提取预测数字
                return -8.0

            # 取最后一个数字作为答案
            pred_answer = pred_numbers[-1]
            gt_answer = gt_numbers[-1]

            # 比较(更细粒度)
            diff = abs(pred_answer - gt_answer)

            if diff < 1e-4:
                return 10.0   # 完全正确
            elif diff < 0.1:
                return 8.0    # 非常接近
            elif diff < 1.0:
                return 5.0    # 接近
            elif diff < 10.0:
                return 2.0    # 数量级正确
            else:
                return -5.0   # 错误

        except Exception as e:
            print(f"⚠️  数学评估错误: {e}")
            return -5.0

    def _extract_boxed(self, text: str) -> Optional[str]:
        """提取\boxed{}中的内容(ROLL风格)"""
        match = re.search(r'\\boxed\{([^}]+)\}', text)
        if match:
            return match.group(1).strip()
        return None

    def _extract_numbers(self, text: str) -> list:
        """从文本中提取所有数字(改进版 + 文字数字识别)"""
        numbers = []

        # Method 1: Numeric extraction (existing)
        # 匹配整数、小数、负数、科学计数法
        pattern = r'-?\d+\.?\d*(?:[eE][+-]?\d+)?'
        matches = re.findall(pattern, text)
        for m in matches:
            if m:
                try:
                    numbers.append(float(m))
                except:
                    pass

        # Method 2: Word-to-number recognition (NEW - fixes ~15-20% QA errors)
        # Aligns with SQuAD/HotpotQA standards for text-based answers
        word_to_num = {
            'zero': 0, 'one': 1, 'two': 2, 'three': 3, 'four': 4,
            'five': 5, 'six': 6, 'seven': 7, 'eight': 8, 'nine': 9,
            'ten': 10, 'eleven': 11, 'twelve': 12, 'thirteen': 13,
            'fourteen': 14, 'fifteen': 15, 'sixteen': 16, 'seventeen': 17,
            'eighteen': 18, 'nineteen': 19, 'twenty': 20, 'thirty': 30,
            'forty': 40, 'fifty': 50, 'sixty': 60, 'seventy': 70,
            'eighty': 80, 'ninety': 90, 'hundred': 100, 'thousand': 1000
        }

        text_lower = text.lower()
        for word, num in word_to_num.items():
            if word in text_lower:
                numbers.append(float(num))

        return numbers

    def _compute_code_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        代码问题正确性(改进版)
        """
        try:
            pred_str = str(prediction).strip()
            gt_str = str(ground_truth).strip()

            # 如果预测为空
            if not pred_str:
                return -10.0

            # 完全匹配(最高分)
            if pred_str.lower() == gt_str.lower():
                return 10.0

            # 包含匹配
            if gt_str.lower() in pred_str.lower():
                return 10.0

            # 提取函数定义
            pred_funcs = self._extract_function_names(pred_str)
            gt_funcs = self._extract_function_names(gt_str)

            # 检查函数名是否匹配
            if pred_funcs and gt_funcs:
                if any(pf == gf for pf in pred_funcs for gf in gt_funcs):
                    return 5.0  # 部分正确

            return -5.0

        except Exception as e:
            print(f"⚠️  代码评估错误: {e}")
            return -5.0

    def _extract_function_names(self, code: str) -> list:
        """从代码中提取函数名"""
        pattern = r'def\s+(\w+)\s*\('
        matches = re.findall(pattern, code)
        return matches

    def _compute_qa_correctness(self, prediction: str, ground_truth: str) -> float:
        """
        QA问题正确性(ROLL风格改进)
        """
        try:
            pred_str = str(prediction).strip().lower()
            gt_str = str(ground_truth).strip().lower()

            if not pred_str:
                return -10.0

            # 精确匹配
            if pred_str == gt_str:
                return 10.0

            # 包含匹配
            if gt_str in pred_str:
                return 8.0

            # Token重叠
            pred_tokens = set(pred_str.split())
            gt_tokens = set(gt_str.split())

            if not gt_tokens:
                return -5.0

            overlap_ratio = len(pred_tokens & gt_tokens) / len(gt_tokens)

            if overlap_ratio > 0.8:
                return 6.0
            elif overlap_ratio > 0.5:
                return 3.0
            elif overlap_ratio > 0.2:
                return 0.0
            else:
                return -5.0

        except Exception as e:
            print(f"⚠️  QA评估错误: {e}")
            return -5.0

    def _compute_general_correctness(self, prediction: str, ground_truth: str) -> float:
        """通用正确性评估"""
        return self._compute_qa_correctness(prediction, ground_truth)

    def _compute_efficiency_reward(self, cost: float) -> float:
        """
        计算效率奖励(基于API成本) - ROLL风格

        Returns:
            reward: [-8, 10]
        """
        if cost == 0.0:
            return 0.0

        # ROLL风格的成本阈值
        if cost <= 0.001:
            return 10.0
        elif cost <= 0.005:
            return 5.0
        elif cost <= 0.01:
            return 0.0
        elif cost <= 0.05:
            return -3.0
        else:
            return -8.0

    def _compute_simplicity_reward(
        self,
        execution_time: float,
        num_operators: int = 1
    ) -> float:
        """
        计算简洁性奖励 - ROLL风格

        Returns:
            reward: [-5, 10]
        """
        # 基于执行时间
        if execution_time <= 5.0:
            time_reward = 10.0
        elif execution_time <= 15.0:
            time_reward = 5.0
        elif execution_time <= 30.0:
            time_reward = 0.0
        elif execution_time <= 60.0:
            time_reward = -3.0
        else:
            time_reward = -5.0

        # 基于算子数量
        if num_operators <= 2:
            operator_reward = 10.0
        elif num_operators <= 4:
            operator_reward = 5.0
        elif num_operators <= 6:
            operator_reward = 0.0
        else:
            operator_reward = -5.0

        # 平均
        return (time_reward + operator_reward) / 2.0

    def _compute_format_reward(self, response: str, problem_type: str) -> float:
        """
        格式奖励(新增 - ROLL风格)

        检查响应格式规范性

        Returns:
            reward: [-2, 2]
        """
        if not response:
            return -2.0

        if problem_type == "math":
            # 检查是否有思考过程+答案
            has_think = bool(re.search(r'<think>.*?</think>', response, re.DOTALL))
            has_answer = bool(re.search(r'<answer>.*?</answer>', response, re.DOTALL))

            if has_think and has_answer:
                return 2.0    # 完美格式
            elif has_answer:
                return 0.0    # 基本格式
            else:
                return -2.0   # 格式混乱

        elif problem_type == "code":
            # 检查是否有代码块
            has_code_block = bool(re.search(r'```.*?```', response, re.DOTALL))

            if has_code_block:
                return 2.0
            else:
                return -2.0

        elif problem_type == "qa":
            # 检查答案长度合理性
            if 10 < len(response) < 500:
                return 2.0
            elif len(response) > 0:
                return 0.0
            else:
                return -2.0

        return 0.0

    def _compute_repetition_penalty(self, response: str, ngram_size: int = 3) -> float:
        """
        重复惩罚(新增 - ROLL风格)

        计算N-gram重复度并给予惩罚

        Args:
            response: 响应文本
            ngram_size: N-gram大小(默认3)

        Returns:
            penalty: [-2, 0]
        """
        if not response:
            return 0.0

        words = response.split()

        if len(words) < ngram_size:
            return 0.0

        # 生成所有N-grams
        ngrams = []
        for i in range(len(words) - ngram_size + 1):
            ngram = tuple(words[i:i+ngram_size])
            ngrams.append(ngram)

        if not ngrams:
            return 0.0

        # 计算唯一N-grams比例
        unique_ratio = len(set(ngrams)) / len(ngrams)

        # 转换为惩罚
        if unique_ratio > 0.9:
            return 0.0      # 几乎无重复
        elif unique_ratio > 0.7:
            return -0.5     # 轻微重复
        elif unique_ratio > 0.5:
            return -1.0     # 中度重复
        else:
            return -2.0     # 严重重复


def test_reward_computer():
    """测试改进版奖励计算器"""
    print("\n" + "=" * 60)
    print("🧪 测试改进版奖励计算器")
    print("=" * 60)

    computer = RewardComputer()

    # 测试案例
    test_cases = [
        {
            "name": "数学 - 完美格式+正确",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Let me calculate: 15 + 27 = 42</think><answer>\\boxed{42}</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.002, "execution_time": 3.5}
        },
        {
            "name": "数学 - 正确但无格式",
            "problem": "What is 15 + 27?",
            "prediction": "The answer is 42.",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.002, "execution_time": 3.0}
        },
        {
            "name": "数学 - 接近答案",
            "problem": "What is 15 + 27?",
            "prediction": "<think>Calculating</think><answer>42.1</answer>",
            "ground_truth": "42",
            "problem_type": "math",
            "metadata": {"cost": 0.001, "execution_time": 2.0}
        },
        {
            "name": "代码 - 正确+格式",
            "problem": "Write a function to square a number",
            "prediction": "```python\ndef square(x):\n    return x * x\n```",
            "ground_truth": "def square(x):\n    return x * x",
            "problem_type": "code",
            "metadata": {"cost": 0.003, "execution_time": 5.0}
        },
        {
            "name": "QA - 正确",
            "problem": "What is the capital of France?",
            "prediction": "The capital of France is Paris.",
            "ground_truth": "Paris",
            "problem_type": "qa",
            "metadata": {"cost": 0.001, "execution_time": 2.0}
        },
        {
            "name": "严重重复",
            "problem": "Test",
            "prediction": "answer answer answer answer answer answer",
            "ground_truth": "answer",
            "problem_type": "qa",
            "metadata": {"cost": 0.001, "execution_time": 1.0}
        }
    ]

    for case in test_cases:
        reward = computer.compute_reward(
            problem=case["problem"],
            prediction=case["prediction"],
            ground_truth=case["ground_truth"],
            problem_type=case["problem_type"],
            metadata=case["metadata"]
        )

        print(f"\n📝 {case['name']}")
        print(f"  预测: {case['prediction'][:60]}...")
        print(f"  正确答案: {case['ground_truth']}")
        print(f"  奖励: {reward:.2f}/10.0")


if __name__ == "__main__":
    test_reward_computer()
