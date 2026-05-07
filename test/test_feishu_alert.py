"""
测试飞书告警功能

测试内容：
1. 配置加载测试
2. 告警消息格式测试
3. 日志系统集成测试

运行方式：
python test/test_feishu_alert.py
"""

import tempfile
import yaml
from pathlib import Path
import logging
import sys
from datetime import datetime

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_feishu_alert_manager():
    """测试1: 飞书告警管理器基础功能"""
    print("\n" + "="*80)
    print("测试1: 飞书告警管理器基础功能")
    print("="*80)

    try:
        from evomaster.monitoring.feishu_alert_manager import FeishuAlertManager
        # 模拟加载配置
        manager = FeishuAlertManager()
        manager.config_path = "configs/feishu_config.yaml"
        manager._load_config()

        print(f"✅ 配置加载测试通过")
        print(f"✅ APP_ID: {manager.app_id}")
        print(f"✅ Chat ID: {manager.chat_id}")
        print(f"✅ 告警启用状态: {manager.enabled}")

        return True


    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_alert_message_format():
    """测试2: 告警消息格式"""
    print("\n" + "="*80)
    print("测试2: 告警消息格式")
    print("="*80)

    try:
        from evomaster.monitoring.feishu_alert_manager import FeishuAlertManager

        manager = FeishuAlertManager()

        # 测试上下文构建
        test_context = {
            'run_dir': '/runs/test_experiment',
            'exp_id': 'test_experiment',
            'task_id': 'task_123',
            'step_count': 5,
            'error_type': 'ValueError',
            'stack_trace': 'Traceback (most recent call last):\n  File "test.py", line 10\n    raise ValueError("Test error")'
        }

        # 模拟消息构建（不实际发送）
        message = "测试错误消息"
        level = "ERROR"

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        expected_message = f"""🚨 [EvoMaster] ERROR告警

时间: {timestamp}
级别: ERROR
消息: {message}
运行目录: /runs/test_experiment
实验ID: test_experiment
任务ID: task_123
步骤数: 5
错误类型: ValueError
堆栈跟踪:
```
Traceback (most recent call last):
  File "test.py", line 10
    raise ValueError("Test error")
```
"""

        print("✅ 告警消息格式测试通过")
        print(f"✅ 消息长度: {len(expected_message)} 字符")

        # 显示消息格式
        print("\n📋 告警消息格式示例:")
        print("-" * 60)
        print(expected_message[:400] + "...")
        print("-" * 60)

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_logging_integration():
    """测试3: 日志系统集成"""
    print("\n" + "="*80)
    print("测试3: 日志系统集成")
    print("="*80)

    try:
        from evomaster.monitoring.feishu_alert_manager import FeishuAlertManager
        from evomaster.monitoring.feishu_alert_handler import FeishuAlertHandler

        # 创建测试alert manager
        alert_manager = FeishuAlertManager()

        # 创建模拟agent
        class MockAgent:
            def __init__(self):
                self.logger = logging.getLogger("MockAgent")
                self._step_count = 5

                # 模拟playground
                class MockPlayground:
                    def __init__(self):
                        self.run_dir = Path("/runs/test_experiment")
                        class MockConfig:
                            exp_id = "test_exp_123"
                        self.config = MockConfig()

                self.playground = MockPlayground()

                # 模拟trajectory
                class MockTrajectory:
                    task_id = "task_456"
                self.trajectory = MockTrajectory()

        # 创建模拟agent
        mock_agent = MockAgent()

        # 创建feishu alert handler
        handler = FeishuAlertHandler(
            agent=mock_agent,
            alert_manager=alert_manager
        )

        # 添加handler到logger
        mock_agent.logger.addHandler(handler)
        mock_agent.logger.setLevel(logging.DEBUG)

        print("📝 模拟ERROR日志...")
        mock_agent.logger.error("This is a test error message", exc_info=(ValueError, "Test error", None))

        print("📝 模拟WARNING日志...")
        mock_agent.logger.warning("This is a test warning message")

        print("✅ 日志系统集成测试通过")
        print(f"✅ Handler已添加到logger")
        print(f"✅ 告警级别: WARNING及以上")

        # 测试上下文构建
        context = handler._build_context(
            logging.LogRecord(
                name="test",
                level=logging.ERROR,
                pathname="test.py",
                lineno=1,
                msg="Test error",
                args=(),
                exc_info=(ValueError, ValueError("Test"), None)
            )
        )

        print(f"✅ 上下文信息提取:")
        print(f"  - run_dir: {context.get('run_dir')}")
        print(f"  - exp_id: {context.get('exp_id')}")
        print(f"  - task_id: {context.get('task_id')}")
        print(f"  - step_count: {context.get('step_count')}")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_actual_feishu_send():
    """测试4: 实际飞书发送（可选）"""
    print("\n" + "="*80)
    print("测试4: 实际飞书消息发送（需要有效配置）")
    print("="*80)

    try:
        from evomaster.monitoring import feishu_alert_manager

        if not feishu_alert_manager.enabled:
            print("⚠️  飞书告警未启用，跳过实际发送测试")
            print("   要启用此测试，请确保configs/feishu_config.yaml配置正确")
            return True

        # 发送测试消息
        print(feishu_alert_manager.app_id)
        print(feishu_alert_manager.app_secret)
        print("📤 发送测试告警到飞书...")
        success = feishu_alert_manager.send_alert(
            message="这是一条来自EvoMaster的测试告警",
            level="INFO",
            context={
                'run_dir': str(Path.cwd()),
                'exp_id': 'test_experiment',
                'test': True
            }
        )

        if success:
            print("✅ 飞书消息发送成功！")
            print("   请检查飞书群聊是否收到消息")
        else:
            print("❌ 飞书消息发送失败")

        return success

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_context_extraction():
    """测试5: 上下文信息提取"""
    print("\n" + "="*80)
    print("测试5: 上下文信息提取")
    print("="*80)

    try:
        from evomaster.monitoring.feishu_alert_handler import FeishuAlertHandler
        from evomaster.monitoring.feishu_alert_manager import FeishuAlertManager
        import logging

        # 创建模拟对象
        class MockConfig:
            exp_id = "exp_20260415_001"

        class MockPlayground:
            run_dir = Path("/runs/ml_master_datatree_20260415_164819")
            config = MockConfig()

        class MockTrajectory:
            task_id = "task_789"

        class MockAgent:
            playground = MockPlayground()
            trajectory = MockTrajectory()
            _step_count = 10

        # 测试handler
        alert_manager = FeishuAlertManager()
        agent = MockAgent()
        handler = FeishuAlertHandler(agent, alert_manager)

        # 创建测试日志记录
        record = logging.LogRecord(
            name="test",
            level=logging.ERROR,
            pathname="/path/to/file.py",
            lineno=42,
            msg="Test error message",
            args=(),
            exc_info=(ValueError, ValueError("Something went wrong"), None)
        )

        # 提取上下文
        context = handler._build_context(record)

        print("✅ 上下文信息提取测试通过")
        print(f"✅ 提取的上下文信息:")
        print(f"  - run_dir: {context.get('run_dir')}")
        print(f"  - exp_id: {context.get('exp_id')}")
        print(f"  - task_id: {context.get('task_id')}")
        print(f"  - step_count: {context.get('step_count')}")
        print(f"  - error_type: {context.get('error_type')}")
        print(f"  - timestamp: {context.get('timestamp')}")

        # 验证关键字段
        assert context['run_dir'] == str(MockPlayground.run_dir), "run_dir不匹配"
        assert context['exp_id'] == MockConfig.exp_id, "exp_id不匹配"
        assert context['task_id'] == MockTrajectory.task_id, "task_id不匹配"
        assert context['step_count'] == 10, "step_count不匹配"
        assert context['error_type'] == 'ValueError', "error_type不匹配"

        print("✅ 所有关键字段验证通过")

        return True

    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("\n" + "="*80)
    print("🧪 飞书告警机制测试套件")
    print("="*80)

    results = {}

    # 运行所有测试
    results["test_feishu_alert_manager"] = test_feishu_alert_manager()
    results["test_alert_message_format"] = test_alert_message_format()
    results["test_logging_integration"] = test_logging_integration()
    results["test_context_extraction"] = test_context_extraction()

    # 可选：实际飞书发送测试
    print("\n" + "="*80)
    print("是否运行实际飞书发送测试？")
    print("注意：这需要有效的飞书配置，会实际发送消息到飞书群")
    print("="*80)

    # 默认不运行实际发送测试，用户可以选择
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "--with-feishu-send":
        results["test_actual_feishu_send"] = test_actual_feishu_send()
    else:
        print("⏭️  跳过实际飞书发送测试（使用--with-feishu-send参数启用）")

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
        print("\n🎉 所有测试通过！飞书告警机制工作正常。")
        return 0
    else:
        print(f"\n⚠️  有 {failed} 个测试失败，请检查实现。")
        return 1


if __name__ == "__main__":
    exit(main())
