import sys
import os

sys.path.insert(0, os.path.join(os.getcwd(), "agentcodebase"))
sys.path.insert(0, os.path.join(os.getcwd()))

from agentcodebase.codebase.test import APITester

if __name__ == "__main__":
    API_KEY = "${OPENAI_API_KEY}"
    BASE_URL = "${LLM_BASE_URL}"
    MODEL = "volcengine/deepseek-v3-2-251201"

    tester = APITester(
        api_key=API_KEY,
        base_url=BASE_URL,
        config_path="AgentCodeBase/config/config.yaml",
    )
    tester.start_periodic_test(
        interval_seconds=1800, enable_feishu_alert=True, models=MODEL
    )
    # tester.run_test(enable_feishu_alert=True, models="gpt-oss")
