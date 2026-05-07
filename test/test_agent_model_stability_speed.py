#!/usr/bin/env python3
"""Live test for agent model stability and response speed.

This test is network-dependent and disabled by default.
Enable with:
    RUN_LIVE_LLM_TEST=1 uv run python -m unittest discover -s test -p "test_agent_model_stability_speed.py"
"""

from __future__ import annotations

import os
import sys
import time
import unittest
from pathlib import Path


project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    rank = int(round((len(ordered) - 1) * p))
    return ordered[max(0, min(rank, len(ordered) - 1))]


class TestAgentModelStabilitySpeed(unittest.TestCase):
    def test_agent_model_stability_and_speed_live(self):
        try:
            from evomaster.agent.agent import AgentConfig, BaseAgent
            from evomaster.agent.tools.base import ToolRegistry
            from evomaster.utils import LLMConfig, TaskInstance, create_llm
        except Exception as exc:
            self.skipTest(f"Missing runtime dependencies for live test: {exc}")

        if os.getenv("RUN_LIVE_LLM_TEST") != "1":
            self.skipTest("Set RUN_LIVE_LLM_TEST=1 to run live model test.")

        api_key = os.getenv("GPUGEEK_API_KEY")
        if not api_key:
            self.skipTest("GPUGEEK_API_KEY is not set.")

        model = os.getenv("GPUGEEK_MODEL", "Vendor2/Gemini-2.5-Flash")
        # OpenAI-compatible endpoint root, matching your example.
        base_url = os.getenv("GPUGEEK_BASE_URL", "https://api.gpugeek.com/v1")

        rounds = int(os.getenv("LIVE_TEST_ROUNDS", "8"))
        temperature = float(os.getenv("LIVE_TEST_TEMPERATURE", "0.7"))
        max_avg_latency_s = float(os.getenv("LIVE_TEST_MAX_AVG_LATENCY_S", "12"))
        max_p95_latency_s = float(os.getenv("LIVE_TEST_MAX_P95_LATENCY_S", "20"))
        min_success_ratio = float(os.getenv("LIVE_TEST_MIN_SUCCESS_RATIO", "0.9"))
        interval_s = float(os.getenv("LIVE_TEST_INTERVAL_S", "0.2"))

        class _ProbeAgent(BaseAgent):
            def _get_system_prompt(self) -> str:
                return "You are a probe assistant. Reply briefly in plain text."

            def _get_user_prompt(self, task: TaskInstance) -> str:
                return task.description

        llm = create_llm(
            LLMConfig(
                provider="openai",
                model=model,
                api_key=api_key,
                base_url=base_url,
                temperature=temperature,
                max_tokens=4000,
                timeout=60,
                max_retries=2,
                retry_delay=0.8,
            )
        )

        agent = _ProbeAgent(
            llm=llm,
            session=object(),  # no tool/session usage in this probe
            tools=ToolRegistry(),
            config=AgentConfig(max_turns=1),
            enable_tools=False,
        )

        latencies: list[float] = []
        errors: list[str] = []
        success_count = 0

        for i in range(rounds):
            task = TaskInstance(
                task_id=f"live_probe_{i}",
                task_type="llm_probe",
                description=f"Reply only with: PONG-{i}",
            )
            start = time.perf_counter()
            try:
                trajectory = agent.run(task)
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)

                content = ""
                if trajectory.steps:
                    last = trajectory.steps[-1].assistant_message
                    if last and last.content:
                        content = str(last.content).strip()

                if content:
                    success_count += 1
                else:
                    errors.append(f"round={i}: empty content")
            except Exception as exc:
                elapsed = time.perf_counter() - start
                latencies.append(elapsed)
                errors.append(f"round={i}: {type(exc).__name__}: {exc}")

            if i < rounds - 1 and interval_s > 0:
                time.sleep(interval_s)

        success_ratio = success_count / rounds if rounds else 0.0
        avg_latency = sum(latencies) / len(latencies) if latencies else 0.0
        p95_latency = _percentile(latencies, 0.95)

        print(
            "\n[agent-live-test]",
            f"model={model}",
            f"base_url={base_url}",
            f"rounds={rounds}",
            f"success={success_count}/{rounds}",
            f"success_ratio={success_ratio:.3f}",
            f"avg_latency_s={avg_latency:.3f}",
            f"p95_latency_s={p95_latency:.3f}",
            f"errors={errors[:3]}",
        )

        self.assertGreaterEqual(
            success_ratio,
            min_success_ratio,
            f"Success ratio too low: {success_ratio:.3f} < {min_success_ratio:.3f}. errors={errors[:5]}",
        )
        self.assertLessEqual(
            avg_latency,
            max_avg_latency_s,
            f"Average latency too high: {avg_latency:.3f}s > {max_avg_latency_s:.3f}s",
        )
        self.assertLessEqual(
            p95_latency,
            max_p95_latency_s,
            f"P95 latency too high: {p95_latency:.3f}s > {max_p95_latency_s:.3f}s",
        )


if __name__ == "__main__":
    unittest.main()

