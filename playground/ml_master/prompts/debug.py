"""Debug Agent Prompt

This prompt is used to fix buggy solutions.
Matches the original ML-Master debug prompt exactly.
"""


def get_introduction(obfuscate: bool = False, check_format: bool = False) -> str:
    """Get the introduction for debug agent.

    Args:
        obfuscate: If True, use obfuscated (non-Kaggle) version
        check_format: If True, add format checking to introduction

    Returns:
        Introduction string
    """
    if check_format:
        if obfuscate:
            return (
                "You are an expert machine learning engineer attempting a task. "
                "Your previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format,"
                "so based on the information below, you should revise it in order to fix this. "
                "Your response should be an implementation outline in natural language,"
                " followed by a single markdown code block which implements the bugfix/solution."
            )
        return (
            "You are a Kaggle grandmaster attending a competition. "
            "Your previous solution had a bug and/or did not produce a submission.csv, or the generated submission.csv was in an incorrect format,"
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )

    if obfuscate:
        return (
            "You are an expert machine learning engineer attempting a task. "
            "Your previous solution had a bug and/or did not produce a submission.csv, "
            "so based on the information below, you should revise it in order to fix this. "
            "Your response should be an implementation outline in natural language,"
            " followed by a single markdown code block which implements the bugfix/solution."
        )
    return (
        "You are a Kaggle grandmaster attending a competition. "
        "Your previous solution had a bug and/or did not produce a submission.csv, "
        "so based on the information below, you should revise it in order to fix this. "
        "Your response should be an implementation outline in natural language,"
        " followed by a single markdown code block which implements the bugfix/solution."
    )


def get_prompt(
    task_desc: str,
    data_preview: str,
    buggy_code: str,
    execution_output: str,
    impl_guideline: list,
    obfuscate: bool = False,
    check_format: bool = False,
) -> dict:
    """Build the complete debug prompt.

    Args:
        task_desc: Task description
        data_preview: Data preview string
        buggy_code: The buggy implementation code
        execution_output: Execution output showing the error
        impl_guideline: Implementation guidelines
        obfuscate: If True, use obfuscated version
        check_format: If True, indicates format validation failed

    Returns:
        Dictionary with 'introduction' and 'user_prompt' keys
    """
    introduction = get_introduction(obfuscate, check_format)

    # Build user prompt
    user_prompt = f"""# Task description
{task_desc}

# Instructions
## Response format
Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences),
followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric.
There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block.

## Implementation guideline
{chr(10).join(f'- {g}' for g in impl_guideline)}

# Data preview
{data_preview}

# Previous (buggy) implementation
{buggy_code}

# Execution output
{execution_output}"""

    return {
        "introduction": introduction,
        "user_prompt": user_prompt,
    }
