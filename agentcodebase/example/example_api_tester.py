import sys
import os

sys.path.insert(0, os.getcwd())

from codebase import APITester


if __name__ == "__main__":
    tester = APITester()
    tester.run_test(models=["gpt-4o"], enable_feishu_alert=True)
    tester.start_periodic_test(interval_seconds=3, models=["gpt-4o"], enable_feishu_alert=True)
