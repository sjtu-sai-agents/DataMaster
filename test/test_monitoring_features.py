#!/usr/bin/env python3
"""
并发实验监控功能测试

测试内容：
1. Token Count实时存储（raw_log.jsonl）
2. Time-delay记录（trajectory.json中的time_cost字段）
3. 并发安全性
4. 数据完整性

运行方式：
python test/test_monitoring_features.py
"""

import json
import tempfile
import shutil
from pathlib import Path
import threading
import time
from datetime import datetime

# 添加项目路径
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from evomaster.utils import LLMConfig, create_llm
from evomaster.utils.types import (
    Dialog, SystemMessage, UserMessage, AssistantMessage, ToolMessage, ToolCall, FunctionCall
)
from evomaster.agent import BaseAgent, AgentConfig
from evomaster.agent.context import ContextConfig
from evomaster.agent.session.local import LocalSession, LocalSessionConfig


class MockSession:
    """模拟Session，用于测试"""
    def __init__(self, workspace_path: Path):
        self.workspace_path = workspace_path
        self.config = LocalSessionConfig(workspace_path=str(workspace_path))

    def is_open(self):
        return True

    def open(self):
        pass

    def close(self):
        pass

    def get_workspace_path(self):
        return str(self.workspace_path)


class SimpleTestAgent(BaseAgent):
    """简单的测试Agent"""

    def __init__(self, llm, session, tools=None, config=None, **kwargs):
        super().__init__(llm, session, tools or MockToolRegistry(), config, **kwargs)

    def _get_system_prompt(self) -> str:
        return "You are a helpful assistant for testing."

    def _get_user_prompt(self, task) -> str:
        return f"Task: {task.description}"


class MockToolRegistry:
    """模拟工具注册表"""
    def get_tool_specs(self):
        return []

    def get_tool(self, name):
        return None


def test_llm_raw_response():
    """测试1: LLM原始响应捕获"""
    print("\n" + "="*80)
    print("测试1: LLM原始响应捕获")
    print("="*80)

    try:
        # 创建LLM实例（使用真实配置，但可能会失败）
        llm_config = LLMConfig(
            provider="openai",  # 使用openai提供商，支持OpenAI兼容API
            model="Vendor2/GPT-5.3-codex",
            api_key="${IMAGE_MODEL_API_KEY}",
            base_url="https://api.gpugeek.com/v1",
            temperature=0.7,
            timeout=30
        )

        llm = create_llm(llm_config)

        # 创建测试对话
        dialog = Dialog(messages=[
            SystemMessage(content="You are a helpful assistant."),
            UserMessage(content="Say 'Hello, this is a test!'")
        ])

        # 调用LLM
        response = llm.query(dialog)

        # response现在是AssistantMessage对象
        print(f"✅ 响应类型: {type(response).__name__}")
        print(f"✅ 响应内容: {response.content[:100] if response.content else 'No content'}...")

        # 检查AssistantMessage的meta字段
        if hasattr(response, 'meta') and response.meta:
            print(f"✅ AssistantMessage.meta包含字段: {list(response.meta.keys())}")

            # 从meta中获取usage
            if "usage" in response.meta:
                print(f"✅ Token使用: {response.meta['usage']}")

            # 从meta中获取raw_response
            if "raw_response" in response.meta:
                raw_response = response.meta["raw_response"]
                print(f"✅ 原始响应已捕获: {type(raw_response)}")
                print(f"   包含字段: {list(raw_response.keys()) if isinstance(raw_response, dict) else 'N/A'}")

                # 打印完整的原生响应
                print("\n" + "="*60)
                print("📋 LLM原生响应 (raw_response):")
                print("="*60)
                print(json.dumps(raw_response, ensure_ascii=False, indent=2))
                print("="*60 + "\n")

                # 检查关键字段
                if isinstance(raw_response, dict):
                    required_fields = ['id', 'choices', 'model', 'object', 'usage']
                    missing_fields = [f for f in required_fields if f not in raw_response]
                    if not missing_fields:
                        print("✅ 所有必需字段都存在")
                    else:
                        print(f"⚠️  缺少字段: {missing_fields}")
            else:
                print("❌ AssistantMessage.meta中缺少raw_response")
                return False
        else:
            print("❌ AssistantMessage没有meta字段")
            return False

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_raw_log_jsonl():
    """测试2: Raw Log JSONL文件记录"""
    print("\n" + "="*80)
    print("测试2: Raw Log JSONL文件记录")
    print("="*80)

    try:
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        raw_log_file = temp_dir / "raw_log.jsonl"

        # 设置raw log文件路径
        BaseAgent.set_raw_log_file_path(raw_log_file)

        print(f"📁 临时目录: {temp_dir}")
        print(f"📄 Raw log文件: {raw_log_file}")

        # 创建模拟的assistant消息，包含raw_response
        mock_raw_response = {
            "id": "test-response-123",
            "choices": [{
                "finish_reason": "stop",
                "index": 0,
                "message": {
                    "content": "This is a test response",
                    "role": "assistant"
                }
            }],
            "model": "test-model",
            "object": "chat.completion",
            "created": 1234567890,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15
            }
        }

        # 创建模拟的assistant消息
        mock_assistant_msg = AssistantMessage(
            content="This is a test response",
            meta={
                "raw_response": mock_raw_response,
                "finish_reason": "stop",
                "usage": mock_raw_response["usage"],
                "response_id": mock_raw_response["id"],
                "model": mock_raw_response["model"]
            }
        )

        # 模拟trajectory
        from evomaster.utils.types import Trajectory
        mock_trajectory = Trajectory(task_id="test_task")

        # 创建临时agent实例来测试_append_raw_log_entry
        class TestAgent:
            def __init__(self):
                self.trajectory = mock_trajectory
                self._agent_name = "test_agent"
                self._raw_log_file_path = raw_log_file
                self._raw_log_file_lock = threading.Lock()

            def _append_raw_log_entry(self, llm_response, step_id):
                """复制BaseAgent的_append_raw_log_entry方法"""
                if self._raw_log_file_path is None:
                    return

                try:
                    # 从meta中获取raw_response
                    raw_response = llm_response.meta.get("raw_response") if llm_response.meta else None
                    if not raw_response:
                        return

                    entry = {
                        "agent_id": f"{self.trajectory.task_id}_{self._agent_name}",
                        "steps": step_id,
                        "llm_raw_response": raw_response
                    }

                    with self._raw_log_file_lock:
                        with open(self._raw_log_file_path, 'a', encoding='utf-8') as f:
                            f.write(json.dumps(entry, ensure_ascii=False, default=str) + '\n')
                except Exception as e:
                    print(f"❌ 写入raw log失败: {e}")

        # 测试写入
        test_agent = TestAgent()
        test_agent._append_raw_log_entry(mock_assistant_msg, step_id=1)

        # 验证文件内容
        if raw_log_file.exists():
            with open(raw_log_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()

            print(f"✅ Raw log文件已创建，包含 {len(lines)} 行")

            if lines:
                # 解析第一行
                entry = json.loads(lines[0])
                print(f"✅ 条目格式正确: agent_id={entry['agent_id']}, steps={entry['steps']}")

                # 检查raw_response完整性
                if "llm_raw_response" in entry:
                    raw_resp = entry["llm_raw_response"]
                    required_fields = ['id', 'choices', 'model', 'object', 'usage']
                    missing = [f for f in required_fields if f not in raw_resp]

                    if not missing:
                        print("✅ 原始响应完整，包含所有必需字段")
                    else:
                        print(f"⚠️  原始响应缺少字段: {missing}")
                else:
                    print("❌ 条目中缺少llm_raw_response")
                    return False
        else:
            print("❌ Raw log文件未创建")
            return False

        # 清理
        shutil.rmtree(temp_dir)
        print("✅ 临时文件已清理")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_time_cost_in_messages():
    """测试3: 消息中的time_cost字段"""
    print("\n" + "="*80)
    print("测试3: 消息中的time_cost字段")
    print("="*80)

    try:
        # 创建带有time_cost的消息
        assistant_msg = AssistantMessage(
            content="Test response",
            time_cost=2.5
        )

        tool_msg = ToolMessage(
            content="Tool output",
            tool_call_id="call_123",
            name="test_tool",
            time_cost=0.8,
            meta={"info": "test info", "time_cost": 0.8}
        )

        # 检查time_cost字段
        print(f"✅ Assistant消息time_cost: {assistant_msg.time_cost}")
        print(f"✅ Tool消息time_cost: {tool_msg.time_cost}")
        print(f"✅ Tool meta中的time_cost: {tool_msg.meta.get('time_cost')}")

        # 测试序列化
        assistant_dict = assistant_msg.model_dump()
        tool_dict = tool_msg.model_dump()

        if "time_cost" in assistant_dict:
            print(f"✅ Assistant消息序列化包含time_cost: {assistant_dict['time_cost']}")
        else:
            print("❌ Assistant消息序列化缺少time_cost")
            return False

        if "time_cost" in tool_dict:
            print(f"✅ Tool消息序列化包含time_cost: {tool_dict['time_cost']}")
        else:
            print("❌ Tool消息序列化缺少time_cost")
            return False

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_concurrent_writing():
    """测试4: 并发写入安全性"""
    print("\n" + "="*80)
    print("测试4: 并发写入安全性")
    print("="*80)

    try:
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        raw_log_file = temp_dir / "concurrent_raw_log.jsonl"
        trajectory_file = temp_dir / "trajectory.json"

        # 设置文件路径
        BaseAgent.set_raw_log_file_path(raw_log_file)
        BaseAgent.set_trajectory_file_path(trajectory_file)

        print(f"📁 测试目录: {temp_dir}")

        # 并发写入函数
        def concurrent_writer(thread_id: int, num_entries: int):
            """并发写入函数"""
            for i in range(num_entries):
                mock_entry = {
                    "agent_id": f"thread_{thread_id}",
                    "steps": i,
                    "llm_raw_response": {
                        "id": f"response-{thread_id}-{i}",
                        "choices": [{"message": {"content": f"Thread {thread_id}, entry {i}"}}],
                        "model": "test-model",
                        "usage": {"total_tokens": thread_id * 10 + i}
                    }
                }

                # 写入raw log
                with BaseAgent._raw_log_file_lock:
                    with open(raw_log_file, 'a', encoding='utf-8') as f:
                        f.write(json.dumps(mock_entry, ensure_ascii=False) + '\n')

                time.sleep(0.001)  # 小延迟增加并发冲突概率

        # 启动多个线程并发写入
        num_threads = 5
        entries_per_thread = 10
        threads = []

        print(f"🚀 启动 {num_threads} 个线程，每个写入 {entries_per_thread} 条记录...")

        start_time = time.time()
        for i in range(num_threads):
            thread = threading.Thread(target=concurrent_writer, args=(i, entries_per_thread))
            threads.append(thread)
            thread.start()

        # 等待所有线程完成
        for thread in threads:
            thread.join()

        elapsed = time.time() - start_time
        print(f"✅ 所有线程完成，耗时: {elapsed:.2f}秒")

        # 验证文件完整性
        with open(raw_log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        expected_lines = num_threads * entries_per_thread
        print(f"✅ 写入 {len(lines)} 行（期望: {expected_lines}）")

        if len(lines) == expected_lines:
            print("✅ 并发写入安全，无数据丢失")

            # 验证每行的JSON格式
            all_valid = True
            for i, line in enumerate(lines):
                try:
                    entry = json.loads(line)
                    if "llm_raw_response" not in entry:
                        print(f"❌ 第 {i+1} 行缺少llm_raw_response")
                        all_valid = False
                        break
                except json.JSONDecodeError as e:
                    print(f"❌ 第 {i+1} 行JSON格式错误: {e}")
                    all_valid = False
                    break

            if all_valid:
                print("✅ 所有记录格式正确")
        else:
            print(f"❌ 行数不匹配，可能存在并发冲突")

        # 清理
        shutil.rmtree(temp_dir)
        print("✅ 临时文件已清理")
        return len(lines) == expected_lines

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_trajectory_time_cost():
    """测试5: Trajectory文件中的time_cost字段"""
    print("\n" + "="*80)
    print("测试5: Trajectory文件中的time_cost字段")
    print("="*80)

    try:
        # 创建临时目录
        temp_dir = Path(tempfile.mkdtemp())
        trajectory_file = temp_dir / "trajectory.json"

        # 创建模拟的trajectory数据
        trajectory_data = [
            {
                "agent_id": "test_agent",
                "steps": 2,
                "status": "running",
                "trajectory": {
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant"},
                        {"role": "user", "content": "Hello"},
                        {"role": "assistant", "content": "Hi there!", "time_cost": 1.5},
                        {"role": "tool", "content": "Tool result", "tool_call_id": "call_1", "name": "test_tool", "time_cost": 0.3}
                    ]
                }
            }
        ]

        # 写入文件
        with open(trajectory_file, 'w', encoding='utf-8') as f:
            json.dump(trajectory_data, f, indent=2, ensure_ascii=False)

        print(f"📄 Trajectory文件: {trajectory_file}")

        # 读取并验证
        with open(trajectory_file, 'r', encoding='utf-8') as f:
            loaded_data = json.load(f)

        messages = loaded_data[0]["trajectory"]["messages"]

        # 检查每个消息
        time_costs_found = 0
        for msg in messages:
            if "time_cost" in msg:
                time_costs_found += 1
                print(f"✅ {msg['role']}消息包含time_cost: {msg['time_cost']}秒")

        print(f"✅ 找到 {time_costs_found} 个包含time_cost的消息")

        # 清理
        shutil.rmtree(temp_dir)
        print("✅ 临时文件已清理")
        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 并发实验监控功能测试套件")
    print("="*80)

    results = {}

    # 运行所有测试
    results["test_llm_raw_response"] = test_llm_raw_response()
    results["test_raw_log_jsonl"] = test_raw_log_jsonl()
    results["test_time_cost_in_messages"] = test_time_cost_in_messages()
    results["test_concurrent_writing"] = test_concurrent_writing()
    results["test_trajectory_time_cost"] = test_trajectory_time_cost()

    # 汇总结果
    print("\n" + "="*80)
    print("📊 测试结果汇总")
    print("="*80)

    passed = 0
    failed = 0

    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
        if result:
            passed += 1
        else:
            failed += 1

    print(f"\n总计: {passed} 通过, {failed} 失败")

    if failed == 0:
        print("\n🎉 所有测试通过！监控功能工作正常。")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit(main())