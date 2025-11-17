#!/usr/bin/env python3
"""
RL工作流生成器 - 使用RL训练的Qwen2.5-7B生成优化的工作流
"""
import torch
import json
import ast
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, LoraConfig, get_peft_model
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import sys
import os

class RLWorkflowGenerator:
    """使用RL训练的Qwen2.5-7B生成优化的工作流"""

    def __init__(
        self,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        lora_checkpoint: Optional[str] = None,
        device_ids: List[int] = [2, 3],
        operator_descriptions_path: Optional[str] = None,
        config: Optional[Dict] = None
    ):
        """
        Args:
            base_model: 基座模型路径
            lora_checkpoint: LoRA检查点路径（None表示使用基座模型）
            device_ids: 使用的GPU ID列表
            operator_descriptions_path: AFlow算子描述文件路径
            config: 额外配置
        """
        self.base_model = base_model
        self.lora_checkpoint = lora_checkpoint
        self.device_ids = device_ids
        self.device = f"cuda:{device_ids[0]}" if torch.cuda.is_available() else "cpu"
        self.config = config or {}

        # 设置CUDA设备
        if torch.cuda.is_available():
            os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(map(str, device_ids))

        print(f"🔧 初始化RL工作流生成器")
        print(f"  设备: {self.device}")
        print(f"  GPU: {device_ids}")

        # 加载tokenizer
        print(f"📥 加载tokenizer: {base_model}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        # 设置pad_token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # 加载模型
        print(f"📥 加载基座模型: {base_model}")
        self.model = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.bfloat16,
            device_map={"": self.device},
            trust_remote_code=True
        )

        # 加载LoRA权重（如果有）
        if lora_checkpoint:
            print(f"📥 加载LoRA检查点: {lora_checkpoint}")
            self.model = PeftModel.from_pretrained(self.model, lora_checkpoint)
            self.model.eval()

        # 加载算子描述
        self.operator_descriptions = self._load_operator_descriptions(operator_descriptions_path)

        print(f"✅ RL工作流生成器初始化完成")

    def _load_operator_descriptions(self, descriptions_path: Optional[str]) -> Dict:
        """加载AFlow算子描述"""
        if descriptions_path and Path(descriptions_path).exists():
            with open(descriptions_path, 'r') as f:
                return json.load(f)

        # 默认算子描述
        return {
            "Custom": {
                "description": "Generates anything based on customized input and instruction.",
                "interface": "custom(input: str, instruction: str) -> dict with key 'response'"
            },
            "AnswerGenerate": {
                "description": "Generates step-by-step reasoning and final answer.",
                "interface": "answer_generate(input: str) -> dict with keys 'thought' and 'answer'"
            },
            "Programmer": {
                "description": "Automatically writes and executes Python code.",
                "interface": "programmer(problem: str, analysis: str = 'None') -> dict with keys 'code' and 'output'"
            },
            "ScEnsemble": {
                "description": "Uses self-consistency to select the most frequent solution.",
                "interface": "sc_ensemble(solutions: List[str], problem: str) -> dict with key 'response'"
            },
            "Review": {
                "description": "Reviews and provides feedback on a solution.",
                "interface": "review(problem: str, solution: str) -> dict with keys 'review_result' and 'feedback'"
            },
            "Revise": {
                "description": "Revises solution based on feedback.",
                "interface": "revise(problem: str, solution: str, feedback: str) -> dict with key 'solution'"
            }
        }

    def _build_generation_prompt(self, problem: str, problem_type: str) -> str:
        """构建提示词，要求生成JSON格式（prompts + graph_code）- 借鉴AFlow风格"""

        prompt = f"""You must generate EXACTLY ONE valid JSON object. Do not generate examples, explanations, or multiple JSONs.

TASK: Solve this {problem_type} problem: {problem}

OUTPUT FORMAT (JSON):
{{
    "prompts": {{
        "OperatorName": "instruction string to optimize for this specific problem"
    }},
    "graph_code": "Python Workflow class code..."
}}

CRITICAL REQUIREMENTS:
1. The "prompts" field contains instruction strings for each operator you use
2. These prompts will be learned by RL - make them problem-specific and effective
3. The "graph_code" should reference prompts via self.prompts["OperatorName"]

Available Operators:

1. Custom(llm) - Most flexible, for any custom task
   Call: await self.custom(input=str, instruction=str)
   Returns: {{'response': str}}

2. AnswerGenerate(llm) - Step-by-step reasoning (NO instruction parameter!)
   Call: await self.answer_generate(input=str)
   Returns: {{'thought': str, 'answer': str}}

3. Programmer(llm) - Auto-generate and execute Python code
   Call: await self.programmer(problem=str, analysis=str)
   Returns: {{'code': str, 'output': str}}

4. Review(llm) - Reviews and provides feedback
   Call: await self.review(problem=str, solution=str)
   Returns: {{'review_result': str, 'feedback': str}}

5. Revise(llm) - Revises solution based on feedback
   Call: await self.revise(problem=str, solution=str, feedback=str)
   Returns: {{'solution': str}}

EXAMPLE OUTPUT:
{{
    "prompts": {{
        "Custom": "用代数方法一步步解决这个数学问题，最后用boxed格式给出答案"
    }},
    "graph_code": "import workspace.{problem_type}.workflows.template.operator as operator\\nfrom scripts.async_llm import create_llm_instance\\nfrom scripts.evaluator import DatasetType\\n\\nclass Workflow:\\n    def __init__(self, name: str, llm_config, dataset: DatasetType):\\n        self.name = name\\n        self.dataset = dataset\\n        self.llm = create_llm_instance(llm_config)\\n        self.custom = operator.Custom(self.llm)\\n        self.prompts = None  # Will be injected at runtime\\n\\n    async def __call__(self, problem: str):\\n        solution = await self.custom(input=problem, instruction=self.prompts['Custom'])\\n        return solution['response'], self.llm.get_usage_summary()['total_cost']"
}}

Generate the JSON:"""

        return prompt

    def generate_workflow(
        self,
        problem: str,
        problem_type: str = "math",
        temperature: float = 0.7,
        max_new_tokens: int = 2048,
        return_full_output: bool = False
    ) -> Dict:
        """
        生成优化的工作流

        Args:
            problem: 问题文本
            problem_type: 问题类型 (math/code/qa)
            temperature: 采样温度
            max_new_tokens: 最大生成token数
            return_full_output: 是否返回完整输出

        Returns:
            {
                "workflow_code": "Python代码",
                "valid": bool,
                "error": Optional[str],
                "metadata": {...}
            }
        """

        # 构建提示词
        prompt = self._build_generation_prompt(problem, problem_type)

        # Tokenize
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)

        # 生成 (优化参数防止重复)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=self.config.get('top_p', 0.95),
                top_k=self.config.get('top_k', 50),
                repetition_penalty=self.config.get('repetition_penalty', 1.2),  # 防止重复生成
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )

        # 解码
        generated_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )

        # 解析输出（期望JSON格式）
        workflow_spec, is_valid, error = self._parse_workflow_output(generated_text, problem_type)

        # 返回完整的workflow_spec（包含prompts和graph_code）
        result = workflow_spec.copy()
        result.update({
            "valid": is_valid,
            "error": error,
            "metadata": {
                "problem": problem,
                "problem_type": problem_type,
                "temperature": temperature,
                "tokens_generated": outputs.shape[1] - inputs['input_ids'].shape[1]
            }
        })

        if return_full_output:
            result["full_output"] = generated_text
            result["prompt"] = prompt

        return result

    def _parse_workflow_output(self, generated_text: str, problem_type: str) -> Tuple[Dict, bool, Optional[str]]:
        """解析生成的文本，提取并验证工作流规范（JSON格式）"""

        # DEBUG: 打印 Qwen 生成的原始文本
        print(f"\n{'='*60}")
        print(f"🔍 DEBUG: Qwen 生成的原始文本 (完整):")
        print(f"{'='*60}")
        print(generated_text)
        print(f"{'='*60}\n")

        # 使用括号计数法提取第一个完整的JSON对象
        json_start = generated_text.find("{")

        if json_start != -1:
            # 从第一个'{'开始计数，找到匹配的'}'
            bracket_count = 0
            json_end = -1

            for i in range(json_start, len(generated_text)):
                if generated_text[i] == '{':
                    bracket_count += 1
                elif generated_text[i] == '}':
                    bracket_count -= 1
                    if bracket_count == 0:
                        json_end = i
                        break

            if json_end != -1:
                json_text = generated_text[json_start:json_end+1]
                print(f"✅ 使用括号计数法提取JSON (长度: {len(json_text)} 字符)")

        if json_start != -1 and json_end != -1 and json_end > json_start:
            try:
                # 解析JSON
                workflow_spec = json.loads(json_text)

                # 验证必需字段
                if "prompts" not in workflow_spec or "graph_code" not in workflow_spec:
                    print(f"⚠️  JSON缺少必需字段 (prompts/graph_code)，使用默认工作流")
                    return self._get_default_workflow(problem_type), False, "Missing required fields in JSON"

                # 验证graph_code的语法
                try:
                    ast.parse(workflow_spec["graph_code"])
                    is_valid = True
                    error = None
                except SyntaxError as e:
                    print(f"⚠️  graph_code语法错误: {e}，使用默认工作流")
                    return self._get_default_workflow(problem_type), False, f"Syntax error in graph_code: {str(e)}"

                print(f"✅ 成功解析JSON工作流规范")
                print(f"  Prompts: {list(workflow_spec['prompts'].keys())}")
                return workflow_spec, is_valid, error

            except json.JSONDecodeError as e:
                print(f"⚠️  JSON解析失败: {e}")

        # JSON解析失败，尝试提取纯代码（向后兼容）
        print(f"⚠️  未找到有效JSON，尝试提取纯代码...")
        code_start = generated_text.find("```python")
        if code_start == -1:
            code_start = generated_text.find("class Workflow:")
            if code_start == -1:
                print(f"⚠️  未找到代码，使用默认工作流")
                return self._get_default_workflow(problem_type), False, "No valid JSON or code found in output"
            code = generated_text[code_start:]
        else:
            code_start += len("```python\n")
            code_end = generated_text.find("```", code_start)
            code = generated_text[code_start:code_end] if code_end != -1 else generated_text[code_start:]

        code = code.strip()

        # 验证语法并包装为workflow_spec
        try:
            ast.parse(code)
            # 从代码中提取默认prompts（简化处理）
            workflow_spec = {
                "prompts": {"Custom": "Solve this problem step by step."},
                "graph_code": code
            }
            print(f"✅ 使用纯代码模式（向后兼容）")
            return workflow_spec, True, None
        except SyntaxError as e:
            print(f"⚠️  代码语法错误: {e}，使用默认工作流")
            return self._get_default_workflow(problem_type), False, f"Syntax error: {str(e)}"

    def _get_default_workflow(self, problem_type: str = "math") -> Dict:
        """默认工作流（当生成失败时），返回dict格式"""
        graph_code = f"""import workspace.{problem_type}.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)
        self.prompts = None  # Will be injected at runtime

    async def __call__(self, problem: str):
        instruction = self.prompts.get("Custom", "Solve this problem step by step.") if self.prompts else "Solve this problem step by step."
        solution = await self.custom(input=problem, instruction=instruction)
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

        return {
            "prompts": {
                "Custom": "Solve this problem step by step."
            },
            "graph_code": graph_code
        }


def test_generator():
    """测试生成器"""
    print("\n" + "=" * 60)
    print("🧪 测试RL工作流生成器")
    print("=" * 60)

    # 注意：这需要Qwen模型，如果没有下载会很慢
    generator = RLWorkflowGenerator(
        base_model="Qwen/Qwen2.5-7B-Instruct",
        device_ids=[2, 3],
        operator_descriptions_path="/home/yijia/.claude/11/AFlow/workspace/MATH/workflows/template/operator.json"
    )

    # 测试问题
    test_problem = "What is 15 + 27?"

    print(f"\n📝 测试问题: {test_problem}")

    # 生成工作流
    result = generator.generate_workflow(
        problem=test_problem,
        problem_type="math",
        temperature=0.7,
        max_new_tokens=1024
    )

    print(f"\n✅ 生成结果:")
    print(f"  有效性: {result['valid']}")
    if result['error']:
        print(f"  错误: {result['error']}")

    print(f"\n📄 生成的工作流代码:")
    print(result['workflow_code'])


if __name__ == "__main__":
    test_generator()
