"""Review Agent Prompt

This prompt is used to evaluate code execution results.
Matches the original ML-Master review prompt exactly.
"""


def get_introduction(obfuscate: bool = False) -> str:
    """Get the introduction for review agent.

    Args:
        obfuscate: If True, use obfuscated (non-Kaggle) version

    Returns:
        Introduction string
    """
    if obfuscate:
        return (
            "You are an expert machine learning engineer attempting a task. "
            "You have written code to solve this task and now need to evaluate the output of the code execution. "
            "You should determine if there were any bugs as well as report the empirical findings."
        )
    return (
        "You are a Kaggle grandmaster attending a competition. "
        "You have written code to solve this task and now need to evaluate the output of the code execution. "
        "You should determine if there were any bugs as well as report the empirical findings."
    )


def get_prompt(
    task_desc: str,
    code: str,
    execution_output: str,
) -> dict:
    """Build the complete review prompt.

    Args:
        task_desc: Task description
        code: The executed code
        execution_output: Execution output

    Returns:
        Dictionary with system prompt content
    """
    introduction = get_introduction()

    prompt = {
        "Introduction": introduction,
        "Task description": task_desc,
        "Implementation": code,
        "Execution output": execution_output,
    }

    return prompt


def get_evaluation_instructions() -> str:
    """Get the evaluation instructions for JSON output.

    Returns:
        Instructions string for JSON format
    """
    return """
Please evaluate the code execution output and provide your analysis in the following JSON format:

```json
{
    "is_bug": true or false,
    "has_csv_submission": true or false,
    "summary": "brief 2-3 sentence summary of empirical findings",
    "metric": numeric value or null,
    "lower_is_better": true or false
}
```

Where:
- is_bug: true if the execution failed or had bugs, false otherwise
- has_csv_submission: true if predictions were saved to ./submission/submission.csv (or submission_<hash>.csv), false otherwise
- summary: 2-3 sentences describing findings or mentioning bugs
- metric: the validation metric value if successful, null if failed
- lower_is_better: true for metrics like MSE/RMSE (lower is better), false for metrics like accuracy/AUC (higher is better)
"""
