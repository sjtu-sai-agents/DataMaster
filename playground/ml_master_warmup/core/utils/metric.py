"""指标解析与安全兜底逻辑。"""

import json
import re
from typing import Any

from .runtime import extract_json_code


def extract_metric_from_stdout(text: str) -> float | None:
    """从 stdout 中提取 metric 值（如 AUC、accuracy 等）"""
    if not text:
        return None

    # 匹配常见的 metric 模式
    patterns = [
        r"Validation AUC[:\s]+([0-9.]+)",
        r"Validation AUC Score[:\s]+([0-9.]+)",
        r"AUC[:\s]+([0-9.]+)",
        r"auc[:\s]+([0-9.]+)",
        r"Accuracy[:\s]+([0-9.]+)",
        r"accuracy[:\s]+([0-9.]+)",
        r"Final validation AUC[:\s]+([0-9.]+)",
        r"ROC AUC[:\s]+([0-9.]+)",
        r"roc_auc_score[:\s]+([0-9.]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except (ValueError, IndexError):
                continue

    return None


def parse_metric_content(text: str) -> dict[str, Any]:
    """尝试解析纯 JSON；失败则从 stdout 提取 metric。"""
    if not text:
        return {"metric": None, "is_bug": True, "error": "empty metric text"}

    cleaned = text.strip()

    # 首先尝试解析 JSON
    try:
        cleaned = extract_json_code(cleaned)
        data = json.loads(cleaned)
        if isinstance(data, dict):
            metric_val = data.get("metric")
            # 检查 metric 是否是有效数值
            if metric_val is not None and isinstance(metric_val, (int, float)):
                return data
            # 如果 metric 是字符串 "metric" 或其他无效值，尝试从 stdout 提取
            if isinstance(metric_val, str):
                # 尝试从原始 text 中提取实际 metric
                extracted = extract_metric_from_stdout(text)
                if extracted is not None:
                    data["metric"] = extracted
                    return data
    except Exception as e:  # noqa: BLE001
        pass

    # 如果 JSON 解析失败或 metric 无效，尝试直接从 stdout 提取
    extracted = extract_metric_from_stdout(text)
    if extracted is not None:
        return {
            "metric": extracted,
            "lower_is_better": False,
            "is_bug": False,
            "has_submission": True,
            "summary": "Extracted from stdout"
        }

    return {"metric": None, "is_bug": True, "error": "metric json parse failed"}
