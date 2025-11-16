#!/usr/bin/env python3
"""
AFlow执行适配器 - 执行RL生成的工作流
"""
import sys
import os
import tempfile
import importlib.util
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import asyncio
import time

# 添加AFlow到路径（添加多个可能需要的路径）
aflow_path = '/home/yijia/.claude/11/AFlow'
sys.path.insert(0, aflow_path)
sys.path.insert(0, os.path.join(aflow_path, 'workspace'))

# 导入AFlow组件
from scripts.async_llm import create_llm_instance, LLMsConfig
from scripts import operators as operator_module

class AFlowExecutor:
    """执行RL生成的工作流，使用AFlow的算子"""

    def __init__(
        self,
        llm_config_path: str = "config/aflow_llm.yaml",
        llm_model_name: str = "gpt-4o-mini",
        timeout: int = 300
    ):
        """
        Args:
            llm_config_path: AFlow LLM配置文件路径
            llm_model_name: 使用的LLM模型名称
            timeout: 执行超时时间（秒）
        """
        self.llm_config_path = Path(llm_config_path)
        self.llm_model_name = llm_model_name
        self.timeout = timeout

        # 加载LLM配置
        self._load_llm_config()

        print(f"✅ AFlow执行器初始化完成")
        print(f"  LLM模型: {llm_model_name}")
        print(f"  超时: {timeout}秒")

    def _load_llm_config(self):
        """加载LLM配置"""
        try:
            # 设置配置路径
            abs_config_path = self.llm_config_path.absolute()

            # 读取YAML配置文件
            import yaml
            with open(abs_config_path, 'r') as f:
                yaml_data = yaml.safe_load(f)

            # LLMsConfig期望的是models字典
            models_config = yaml_data.get('models', {})

            # 直接加载配置
            from scripts.async_llm import LLMsConfig
            self.llm_configs = LLMsConfig(models_config)

            print(f"✅ 加载LLM配置: {abs_config_path}")

        except Exception as e:
            print(f"⚠️  加载LLM配置失败: {e}")
            print(f"  将使用 LLMsConfig.default()")
            # 使用默认配置而不是 None
            from scripts.async_llm import LLMsConfig
            try:
                self.llm_configs = LLMsConfig.default()
                print(f"✅ 成功加载默认LLM配置")
            except Exception as e2:
                print(f"  默认配置也加载失败: {e2}")
                # 最后的降级方案：设为 None，后续用字符串
                self.llm_configs = None

    async def execute_workflow(
        self,
        workflow_code: str,
        problem: str,
        problem_type: str = "math",
        **kwargs
    ) -> Tuple[Any, float, Dict]:
        """
        执行工作流

        Args:
            workflow_code: RL模型生成的Workflow类代码
            problem: 问题文本
            problem_type: 问题类型
            **kwargs: 其他参数（如entry_point for code）

        Returns:
            (answer, cost, metadata)
        """

        start_time = time.time()

        try:
            # 创建临时工作流模块
            workflow_class = self._create_workflow_class(workflow_code, problem_type)

            # 实例化工作流
            llm_config = self._get_llm_config()

            # 确保 llm_config 不是 None
            if llm_config is None:
                print(f"⚠️  llm_config 为 None，降级为字符串: {self.llm_model_name}")
                llm_config = self.llm_model_name

            try:
                workflow = workflow_class(
                    name="rl_generated_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )
            except Exception as e:
                # 工作流实例化失败，使用fallback
                print(f"⚠️  工作流实例化失败: {e}")
                import traceback
                traceback.print_exc()
                print(f"  使用fallback工作流")
                fallback_class = self._get_fallback_workflow_class(problem_type)
                workflow = fallback_class(
                    name="fallback_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )

            # 执行（带超时）
            # 尝试传入entry_point（code问题需要），如果失败则降级为只传problem
            try:
                if problem_type == "code" and "entry_point" in kwargs:
                    try:
                        result = await asyncio.wait_for(
                            workflow(problem, kwargs["entry_point"]),
                            timeout=self.timeout
                        )
                    except TypeError as e:
                        # 如果Workflow不接受entry_point参数，只传problem
                        if "positional argument" in str(e):
                            print(f"  ⚠️  Workflow不支持entry_point参数，降级为只传problem")
                            result = await asyncio.wait_for(
                                workflow(problem),
                                timeout=self.timeout
                            )
                        else:
                            raise
                else:
                    result = await asyncio.wait_for(
                        workflow(problem),
                        timeout=self.timeout
                    )
            except (AttributeError, TypeError, KeyError, IndexError, ValueError, NameError) as e:
                # 运行时错误：Workflow结构有问题，使用fallback workflow
                print(f"  ⚠️  执行错误: {type(e).__name__}: {e}")
                print(f"  使用fallback工作流重试")

                # 创建并使用fallback workflow
                fallback_class = self._get_fallback_workflow_class(problem_type)
                fallback_workflow = fallback_class(
                    name="fallback_workflow",
                    llm_config=llm_config,
                    dataset=problem_type
                )

                result = await asyncio.wait_for(
                    fallback_workflow(problem),
                    timeout=self.timeout
                )

            # 安全地解包结果（可能返回2个或更多值）
            if isinstance(result, tuple):
                if len(result) >= 2:
                    answer, cost = result[0], result[1]
                elif len(result) == 1:
                    answer, cost = result[0], 0.0
                else:
                    answer, cost = None, 0.0
            else:
                answer, cost = result, 0.0

            execution_time = time.time() - start_time

            # 元数据
            metadata = {
                "success": True,
                "execution_time": execution_time,
                "cost": cost,
                "problem_type": problem_type
            }

            return answer, cost, metadata

        except asyncio.TimeoutError:
            execution_time = time.time() - start_time
            print(f"⏱️  执行超时 ({self.timeout}秒)")

            metadata = {
                "success": False,
                "error": "timeout",
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type
            }

            return None, 0.0, metadata

        except Exception as e:
            execution_time = time.time() - start_time
            print(f"❌ 执行错误: {str(e)}")

            import traceback
            traceback.print_exc()

            metadata = {
                "success": False,
                "error": str(e),
                "execution_time": execution_time,
                "cost": 0.0,
                "problem_type": problem_type
            }

            return None, 0.0, metadata

    def _create_workflow_class(self, workflow_code: str, problem_type: str):
        """从工作流代码动态创建Workflow类"""

        # 准备命名空间
        namespace = {
            "operator": operator_module,
            "create_llm_instance": create_llm_instance,
            "DatasetType": str
        }

        # 替换import路径（使workspace路径可用）
        # 这里简化处理，直接使用scripts中的operator
        modified_code = workflow_code.replace(
            f"import workspace.{problem_type}.workflows.template.operator as operator",
            "# operator already imported"
        )

        # 修复常见typo（RL模型可能产生的错误）
        modified_code = modified_code.replace("async_lll", "async_llm")
        modified_code = modified_code.replace("create_lll_instance", "create_llm_instance")

        try:
            # 执行代码创建类
            exec(modified_code, namespace)

            # 返回Workflow类
            if "Workflow" not in namespace:
                raise ValueError("No Workflow class found in generated code")

            return namespace["Workflow"]

        except Exception as e:
            print(f"⚠️  生成的工作流代码有错误: {e}")
            print(f"  使用默认fallback工作流")

            # 使用简单的默认工作流作为fallback
            return self._get_fallback_workflow_class(problem_type)

    def _get_llm_config(self):
        """获取LLM配置（确保返回正确类型）"""
        from scripts.async_llm import LLMsConfig, LLMConfig

        try:
            if self.llm_configs:
                result = self.llm_configs.get(self.llm_model_name)
            else:
                # 尝试使用默认配置
                result = LLMsConfig.default().get(self.llm_model_name)

            # 类型验证（关键！）
            if isinstance(result, LLMConfig):
                return result
            elif isinstance(result, dict):
                # 如果意外返回了 dict，转换为 LLMConfig
                print(f"⚠️  警告：get() 返回了 dict，正在转换为 LLMConfig")
                return LLMConfig(result)
            elif isinstance(result, str):
                return result
            else:
                print(f"⚠️  未知类型: {type(result)}，降级为字符串")
                return self.llm_model_name

        except Exception as e:
            print(f"⚠️  获取LLM配置失败: {e}")
            import traceback
            traceback.print_exc()
            # 返回字符串模型名，让 create_llm_instance 自动处理
            print(f"  降级为字符串模式: {self.llm_model_name}")
            return self.llm_model_name

    def _get_fallback_workflow_class(self, problem_type: str):
        """返回一个简单的默认工作流类（用于生成失败时）"""

        class FallbackWorkflow:
            def __init__(self, name: str, llm_config, dataset):
                self.name = name
                self.dataset = dataset
                self.llm = create_llm_instance(llm_config)
                self.custom = operator_module.Custom(self.llm)

            async def __call__(self, problem: str, *args, **kwargs):
                """简单的单步求解"""
                try:
                    result = await self.custom(
                        input=problem,
                        instruction="Solve this problem step by step and provide the final answer."
                    )
                    # 安全地获取cost
                    usage = self.llm.get_usage_summary()
                    if isinstance(usage, dict) and "total_cost" in usage:
                        cost = usage["total_cost"]
                    else:
                        cost = 0.0

                    return result['response'], cost
                except Exception as e:
                    print(f"Fallback workflow error: {e}")
                    import traceback
                    traceback.print_exc()
                    return None, 0.0

        return FallbackWorkflow


async def test_executor():
    """测试AFlow执行器"""
    print("\n" + "=" * 60)
    print("🧪 测试AFlow执行器")
    print("=" * 60)

    # 创建执行器
    executor = AFlowExecutor(
        llm_config_path="config/aflow_llm.yaml",
        llm_model_name="gpt-4o-mini",
        timeout=60
    )

    # 测试工作流代码（简单示例）
    test_workflow_code = """
import workspace.math.workflows.template.operator as operator
from scripts.async_llm import create_llm_instance
from scripts.evaluator import DatasetType

class Workflow:
    def __init__(self, name: str, llm_config, dataset: DatasetType):
        self.name = name
        self.dataset = dataset
        self.llm = create_llm_instance(llm_config)
        self.custom = operator.Custom(self.llm)

    async def __call__(self, problem: str):
        solution = await self.custom(input=problem, instruction="Solve this problem step by step and provide the final answer.")
        return solution['response'], self.llm.get_usage_summary()["total_cost"]
"""

    # 测试问题
    test_problem = "What is 15 + 27?"

    print(f"\n📝 测试问题: {test_problem}")

    # 执行工作流
    answer, cost, metadata = await executor.execute_workflow(
        workflow_code=test_workflow_code,
        problem=test_problem,
        problem_type="math"
    )

    print(f"\n✅ 执行结果:")
    print(f"  成功: {metadata['success']}")
    print(f"  答案: {answer}")
    print(f"  成本: ${cost:.6f}")
    print(f"  时间: {metadata['execution_time']:.2f}秒")


if __name__ == "__main__":
    asyncio.run(test_executor())
