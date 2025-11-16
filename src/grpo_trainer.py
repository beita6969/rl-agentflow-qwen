#!/usr/bin/env python3
"""
GRPO训练器 - 在线学习模式的强化学习训练器
"""
import torch
import torch.nn.functional as F
import asyncio
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import time
import json
import wandb  # ✨ 新增wandb集成

from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from data_manager import DataManager
from rl_workflow_generator import RLWorkflowGenerator
from aflow_executor import AFlowExecutor
from reward_computer import RewardComputer
from gpu_manager import GPUManager


class GRPOTrainer:
    """GRPO训练器：在线学习模式"""

    def __init__(self, config_path: str = "config/training.yaml"):
        """
        Args:
            config_path: 训练配置文件路径
        """
        # 加载配置
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        print("=" * 60)
        print("🚀 初始化GRPO训练器")
        print("=" * 60)

        # GPU管理（使用物理GPU ID）
        physical_gpus = self.config.get('physical_gpus', self.config['device_mapping'])
        self.gpu_manager = GPUManager(
            target_gpus=physical_gpus,
            protected_pids=self.config.get('protected_pids', []),
            auto_clean=True
        )

        # 验证GPU环境
        if not self.gpu_manager.verify_environment():
            raise RuntimeError("GPU环境验证失败")

        # ✨ 初始化wandb
        self._initialize_wandb()

        # 初始化组件
        self._initialize_components()

        print("=" * 60)
        print("✅ GRPO训练器初始化完成")
        print("=" * 60)

    def _initialize_wandb(self):
        """初始化wandb监控"""
        # 从配置或环境变量获取wandb设置
        wandb_config = self.config.get('wandb', {})

        # 设置API key(如果提供的话)
        wandb_api_key = wandb_config.get('api_key', 'b42ca0000cf06f97b05eba34f58823ad5f3122a4')

        # 尝试登录,如果失败则使用offline模式
        try:
            if wandb_api_key and len(wandb_api_key) == 40:
                wandb.login(key=wandb_api_key)
                mode = "online"
            else:
                print("⚠️  wandb API key无效或未提供,使用offline模式")
                mode = "offline"
        except Exception as e:
            print(f"⚠️  wandb登录失败: {e}, 使用offline模式")
            mode = "offline"

        # 初始化wandb run
        wandb.init(
            project=wandb_config.get('project', 'aflow-roll-integration'),
            name=wandb_config.get('run_name', f"grpo-training-{time.strftime('%Y%m%d-%H%M%S')}"),
            mode=mode,  # online或offline
            config={
                # 训练配置
                "base_model": self.config['base_model'],
                "learning_rate": self.config['learning_rate'],
                "batch_size": self.config['rollout_batch_size'],
                "num_sequences": self.config['num_return_sequences_in_group'],
                "max_steps": self.config['max_steps'],
                "lora_rank": self.config['lora_rank'],
                "lora_alpha": self.config['lora_alpha'],
                # 数据配置
                "domain_ratios": self.config['domain_ratios'],
                # 奖励配置
                "reward_weights": self.config.get('reward_weights', {}),
            },
            tags=["grpo", "aflow", "roll", "workflow-generation"],
            notes="GRPO training with improved reward function (ROLL+AgentFlow design)"
        )

        print("\n✅ wandb初始化完成")
        print(f"  模式: {mode}")
        print(f"  项目: {wandb.run.project}")
        print(f"  Run名称: {wandb.run.name}")
        if mode == "online":
            print(f"  Run URL: {wandb.run.url}")
        else:
            print(f"  离线日志: wandb/offline-run-*")

    def _initialize_components(self):
        """初始化所有组件"""

        # 1. 数据管理器
        print("\n📂 初始化数据管理器...")
        self.data_manager = DataManager(
            data_dir=self.config['data_dir'],
            domain_ratios=self.config['domain_ratios']
        )
        self.data_manager.initialize()

        # 2. RL模型（Qwen2.5-7B + LoRA）
        print("\n🤖 加载RL模型...")
        self._load_rl_model()

        # 3. RL工作流生成器（共享已加载的模型）
        print("\n🔧 初始化工作流生成器...")
        self.generator = RLWorkflowGenerator(
            base_model=self.config['base_model'],  # 传递路径用于加载tokenizer
            device_ids=self.config['device_mapping'],
            operator_descriptions_path=self.config.get('aflow_operator_descriptions_path')
        )
        # 共享已加载的模型（避免重复加载）
        self.generator.model = self.model
        self.generator.tokenizer = self.tokenizer

        # 4. AFlow执行器
        print("\n⚙️  初始化AFlow执行器...")
        timeout = self.config.get('execution_timeout', 180)  # 默认180秒
        self.executor = AFlowExecutor(
            llm_config_path=self.config['aflow_config_path'],
            timeout=timeout
        )
        print(f"  执行超时: {timeout}秒")

        # 5. 奖励计算器
        print("\n🎯 初始化奖励计算器...")
        self.reward_computer = RewardComputer(
            reward_weights=self.config.get('reward_weights')
        )

        # 6. 优化器
        print("\n🔬 初始化优化器...")
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config['learning_rate'],
            weight_decay=self.config.get('weight_decay', 0.01)
        )

    def _load_rl_model(self):
        """加载RL模型（Qwen2.5-7B + LoRA）"""
        device = f"cuda:{self.config['device_mapping'][0]}"

        # 加载tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config['base_model'],
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载基座模型
        self.model = AutoModelForCausalLM.from_pretrained(
            self.config['base_model'],
            torch_dtype=torch.bfloat16 if self.config.get('bf16') else torch.float16,
            device_map={"": device},
            trust_remote_code=True
        )

        # 应用LoRA
        if self.config.get('use_lora', True):
            lora_config = LoraConfig(
                r=self.config['lora_rank'],
                lora_alpha=self.config['lora_alpha'],
                target_modules=self.config['lora_target_modules'].split(','),
                lora_dropout=self.config['lora_dropout'],
                bias="none",
                task_type="CAUSAL_LM"
            )
            self.model = get_peft_model(self.model, lora_config)

            print(f"✅ LoRA应用完成")
            self.model.print_trainable_parameters()

    async def train_step(self, step: int) -> Dict:
        """
        单步GRPO训练（在线学习）

        Returns:
            metrics: 训练指标
        """

        # 1. 采样batch
        batch = self.data_manager.sample_batch(
            batch_size=self.config['rollout_batch_size'],
            split="train"
        )

        # 统计
        batch_stats = self.data_manager.get_batch_stats(batch)
        print(f"\n📦 Batch {step}: {len(batch)} 样本, 分布: {batch_stats}")

        # 2. 为每个问题生成K个工作流（GRPO组）
        all_workflows = []
        all_problems = []
        all_answers = []
        all_rewards = []
        all_log_probs = []

        # ✨ 新增：准确率统计
        correctness_scores = []  # 存储所有正确性分数

        num_sequences = self.config['num_return_sequences_in_group']

        for sample in tqdm(batch, desc="生成和执行工作流"):
            problem = sample['problem']
            ground_truth = sample['ground_truth']
            problem_type = sample['problem_type']

            # GRPO组
            group_workflows = []
            group_answers = []
            group_rewards = []
            group_log_probs = []

            for i in range(num_sequences):
                # 生成工作流
                result = self.generator.generate_workflow(
                    problem=problem,
                    problem_type=problem_type,
                    temperature=self.config['generation_config']['temperature']
                )

                workflow_code = result['workflow_code']

                # 计算log概率（旧策略）
                log_prob = await self._compute_log_prob(problem, workflow_code, problem_type)

                # 执行工作流
                try:
                    answer, cost, metadata = await self.executor.execute_workflow(
                        workflow_code=workflow_code,
                        problem=problem,
                        problem_type=problem_type,
                        entry_point=sample.get('entry_point', '')
                    )

                    # 计算奖励
                    if metadata['success']:
                        reward = self.reward_computer.compute_reward(
                            problem=problem,
                            prediction=answer,
                            ground_truth=ground_truth,
                            problem_type=problem_type,
                            metadata=metadata
                        )

                        # ✨ 新增：显式计算并记录正确性
                        correctness = self.reward_computer._compute_correctness_reward(
                            prediction=answer,
                            ground_truth=ground_truth,
                            problem_type=problem_type
                        )
                        correctness_scores.append(correctness)

                        # 判断是否正确（correctness > 5.0 表示接近正确或完全正确）
                        is_correct = correctness >= 5.0
                        status_icon = "✅" if is_correct else "❌"

                        print(f"  {status_icon} 正确性评分: {correctness:.1f}/10.0 | 预测: {str(answer)[:50]} | 真值: {str(ground_truth)[:50]}")
                    else:
                        reward = -10.0  # 执行失败惩罚
                        correctness_scores.append(-10.0)
                        print(f"  ❌ 执行失败 | 真值: {str(ground_truth)[:50]}")

                except Exception as e:
                    print(f"  ⚠️  执行错误: {type(e).__name__}: {e}")
                    import traceback
                    traceback.print_exc()
                    answer = None
                    reward = -10.0
                    correctness_scores.append(-10.0)

                group_workflows.append(workflow_code)
                group_answers.append(answer)
                group_rewards.append(reward)
                group_log_probs.append(log_prob)

            # GRPO关键：组内奖励归一化
            mean_reward = np.mean(group_rewards)
            group_advantages = [r - mean_reward for r in group_rewards]

            # 收集
            all_workflows.extend(group_workflows)
            all_problems.extend([problem] * num_sequences)
            all_answers.extend(group_answers)
            all_rewards.extend(group_advantages)  # 使用优势
            all_log_probs.extend(group_log_probs)

        # 3. 策略梯度更新
        print(f"\n🔄 更新策略...")
        loss, kl_div = await self._update_policy(
            problems=all_problems,
            workflows=all_workflows,
            old_log_probs=all_log_probs,
            advantages=all_rewards,
            problem_types=[s['problem_type'] for s in batch for _ in range(num_sequences)]
        )

        # 4. 指标
        # ✨ 新增：计算准确率统计
        num_correct = sum(1 for score in correctness_scores if score >= 5.0)
        num_total = len(correctness_scores)
        accuracy = (num_correct / num_total * 100) if num_total > 0 else 0.0
        avg_correctness = np.mean(correctness_scores) if correctness_scores else 0.0

        metrics = {
            "step": step,
            "loss": loss,
            "kl_div": kl_div,
            "avg_reward": np.mean(all_rewards),
            "max_reward": np.max(all_rewards),
            "min_reward": np.min(all_rewards),
            "num_samples": len(all_workflows),
            # ✨ 新增准确率指标
            "accuracy": accuracy,
            "num_correct": num_correct,
            "num_total": num_total,
            "avg_correctness_score": avg_correctness
        }

        print(f"\n🎯 准确率统计: {num_correct}/{num_total} = {accuracy:.1f}% (平均正确性评分: {avg_correctness:.2f}/10.0)")

        # ✨ wandb logging
        wandb.log({
            "train/loss": loss,
            "train/kl_div": kl_div,
            "train/avg_reward": np.mean(all_rewards),
            "train/max_reward": np.max(all_rewards),
            "train/min_reward": np.min(all_rewards),
            "train/accuracy": accuracy,
            "train/avg_correctness_score": avg_correctness,
            "train/num_correct": num_correct,
            "train/num_total": num_total,
            "step": step
        })

        return metrics

    async def _compute_log_prob(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（旧策略）"""

        self.model.eval()

        with torch.no_grad():
            # 构建完整文本
            prompt = self.generator._build_generation_prompt(problem, problem_type)
            full_text = prompt + workflow_code

            # Tokenize
            inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

            # 前向传播
            outputs = self.model(**inputs, labels=inputs["input_ids"])

            # 负对数似然 -> log概率
            log_prob = -outputs.loss

            return log_prob.detach().cpu()

    async def _update_policy(
        self,
        problems: List[str],
        workflows: List[str],
        old_log_probs: List[torch.Tensor],
        advantages: List[float],
        problem_types: List[str]
    ) -> Tuple[float, float]:
        """更新策略（GRPO）"""

        self.model.train()

        total_loss = 0.0
        total_kl = 0.0
        num_updates = 0

        # 梯度累积
        grad_accum_steps = self.config.get('gradient_accumulation_steps', 1)

        for i in range(0, len(workflows), grad_accum_steps):
            batch_slice = slice(i, min(i + grad_accum_steps, len(workflows)))

            batch_loss = 0.0
            batch_kl = 0.0

            for j in range(i, min(i + grad_accum_steps, len(workflows))):
                problem = problems[j]
                workflow = workflows[j]
                old_log_prob = old_log_probs[j]
                advantage = advantages[j]
                problem_type = problem_types[j]

                # 计算新log概率
                new_log_prob = await self._compute_log_prob_trainable(problem, workflow, problem_type)

                # 重要性采样比
                ratio = torch.exp(new_log_prob - old_log_prob.to(self.model.device))

                # PPO裁剪损失
                clip_range = self.config['clip_range']
                clipped_ratio = torch.clamp(ratio, 1.0 - clip_range, 1.0 + clip_range)

                advantage_tensor = torch.tensor(advantage, device=self.model.device)
                policy_loss = -torch.min(
                    ratio * advantage_tensor,
                    clipped_ratio * advantage_tensor
                )

                # KL正则化
                if self.config.get('use_kl_loss'):
                    kl_loss = self.config['kl_loss_coef'] * (new_log_prob - old_log_prob.to(self.model.device)).pow(2)
                else:
                    kl_loss = 0.0

                # 总损失
                loss = policy_loss + kl_loss

                # 累积
                batch_loss += loss
                batch_kl += kl_loss if isinstance(kl_loss, torch.Tensor) else 0.0

            # 平均
            batch_loss = batch_loss / min(grad_accum_steps, len(workflows) - i)

            # 反向传播
            batch_loss.backward()

            total_loss += batch_loss.item()
            total_kl += batch_kl.item() if isinstance(batch_kl, torch.Tensor) else batch_kl
            num_updates += 1

            # 优化器步骤
            if (i + grad_accum_steps) % grad_accum_steps == 0:
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('max_grad_norm', 1.0))
                self.optimizer.step()
                self.optimizer.zero_grad()

        avg_loss = total_loss / max(num_updates, 1)
        avg_kl = total_kl / max(num_updates, 1)

        return avg_loss, avg_kl

    async def _compute_log_prob_trainable(
        self,
        problem: str,
        workflow_code: str,
        problem_type: str
    ) -> torch.Tensor:
        """计算工作流的log概率（新策略，可训练）"""

        # 构建完整文本
        prompt = self.generator._build_generation_prompt(problem, problem_type)
        full_text = prompt + workflow_code

        # Tokenize
        inputs = self.tokenizer(full_text, return_tensors="pt").to(self.model.device)

        # 前向传播
        outputs = self.model(**inputs, labels=inputs["input_ids"])

        # 负对数似然 -> log概率
        log_prob = -outputs.loss

        return log_prob

    async def train(self):
        """完整训练循环"""
        print("\n" + "=" * 60)
        print("🎓 开始GRPO训练")
        print("=" * 60)

        max_steps = self.config['max_steps']
        save_every = self.config.get('save_every', 50)
        log_every = self.config.get('log_every', 5)

        for step in range(1, max_steps + 1):
            print(f"\n{'=' * 60}")
            print(f"📍 Step {step}/{max_steps}")
            print(f"{'=' * 60}")

            # 训练步骤
            metrics = await self.train_step(step)

            # 日志
            if step % log_every == 0:
                print(f"\n📊 Metrics:")
                for key, value in metrics.items():
                    print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")

            # 保存检查点
            if step % save_every == 0:
                self.save_checkpoint(step)

        print("\n" + "=" * 60)
        print("✅ 训练完成")
        print("=" * 60)

    def save_checkpoint(self, step: int):
        """保存检查点"""
        checkpoint_dir = Path(self.config['output_dir']) / f"step_{step}"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # 保存LoRA权重
        self.model.save_pretrained(checkpoint_dir)

        print(f"💾 检查点已保存: {checkpoint_dir}")


async def main():
    """主函数"""
    trainer = GRPOTrainer(config_path="config/training.yaml")
    await trainer.train()


if __name__ == "__main__":
    asyncio.run(main())
