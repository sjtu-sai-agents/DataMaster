"""Utility functions for ML-Master playground"""

from .data_preview import generate, generate_for_task
from .response import (
    extract_code,
    extract_text_up_to_code,
    wrap_code,
    extract_review,
    trim_long_string,
    extract_metric_from_output,
)
from .llm_query import plan_and_code_query, query_with_feedback, code_query, _compile_prompt_to_md, extract_after_think, plan_and_code_query_with_agent
from .mcts_utils import linear_decay, exponential_decay, piecewise_decay, dynamic_piecewise_decay
from .server_utils import is_server_online, call_validate, validate_submission_format
from .preproc_data import (
    extract_tar_file,
    extract_zip_file,
    preprocess_data,
    create_directory_structure,
    verify_directory_structure,
)

__all__ = [
    # data_preview
    "generate",
    "generate_for_task",
    # response
    "extract_code",
    "extract_text_up_to_code",
    "wrap_code",
    "extract_review",
    "trim_long_string",
    "extract_metric_from_output",
    # llm_query
    "plan_and_code_query",
    "plan_and_code_query_with_agent"
    "query_with_feedback",
    "code_query",
    "_compile_prompt_to_md",
    "extract_after_think",
    # mcts_utils
    "linear_decay",
    "exponential_decay",
    "piecewise_decay",
    "dynamic_piecewise_decay",
    # server_utils
    "is_server_online",
    "call_validate",
    "validate_submission_format",
    # preproc_data
    "extract_tar_file",
    "extract_zip_file",
    "preprocess_data",
    "create_directory_structure",
    "verify_directory_structure",
]
