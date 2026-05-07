"""Draft Agent Prompt

This prompt is used to generate new initial solutions from the root node.
Matches the original ML-Master draft prompt exactly.
"""

import random


def get_introduction(obfuscate: bool = False) -> str:
    """Get the introduction for draft agent.

    Args:
        obfuscate: If True, use obfuscated (non-Kaggle) version

    Returns:
        Introduction string
    """
    if obfuscate:
        return (
            "You are an expert machine learning engineer attempting a task. "
            "In order to complete this task, you need to come up with an excellent and creative plan "
            "for a solution and then implement this solution in Python. We will now provide a description of the task."
        )
    return (
        "You are a Kaggle grandmaster attending a competition. "
        "In order to win this competition, you need to come up with an excellent and creative plan "
        "for a solution and then implement this solution in Python. We will now provide a description of the task."
    )


def get_prompt(
    task_desc: str,
    memory: str,
    data_preview: str,
    impl_guideline: list,
    environment: dict,
    obfuscate: bool = False,
) -> dict:
    """Build the complete draft prompt.

    Args:
        task_desc: Task description
        memory: Memory of previous solutions
        data_preview: Data preview string
        impl_guideline: Implementation guidelines
        environment: Environment info (installed packages)
        obfuscate: If True, use obfuscated version

    Returns:
        Dictionary with 'introduction' and 'user_prompt' keys
    """
    introduction = get_introduction(obfuscate)

    # Response format
    response_format = {
        "Response format": (
            "Your response should be a brief outline/sketch of your proposed solution in natural language (3-5 sentences), "
            "followed by a single markdown code block (wrapped in ```) which implements this solution and prints out the evaluation metric. "
            "There should be no additional headings or text in your response. Just natural language text followed by a newline and then the markdown code block. "
        )
    }

    # Solution sketch guideline
    solution_guideline = {
        "Solution sketch guideline": [
            "- This first solution design should be relatively simple, without ensembling or hyper-parameter optimization.",
            "- When proposing the design, take the Memory section into account.",
            "- In addition to incorporating the Memory module, it is **crucial** that your proposed solution **is distinctly different from** the existing designs in the Memory section.",
            "- Don't propose the same modelling solution but keep the evaluation the same.",
            "- The solution sketch should be 3-5 sentences.",
            "- Propose an evaluation metric that is reasonable for this task.",
            "- Don't suggest to do EDA.",
            "- The data is already prepared and available in the `./input` directory. There is no need to unzip any files.",
        ],
    }

    # Combine all instructions
    instructions = {**response_format, **solution_guideline, "Implementation guideline": impl_guideline, **environment}

    # Build user prompt
    user_prompt = f"""# Task description
{task_desc}

# Memory
The memory of previous solutions used to solve task is provided below:
{memory}

# Instructions
{_format_instructions(instructions)}

# Data preview
{data_preview}"""

    return {
        "introduction": introduction,
        "user_prompt": user_prompt,
    }


def _format_instructions(instructions: dict) -> str:
    """Format instructions dict to markdown string.

    Args:
        instructions: Instructions dictionary

    Returns:
        Formatted markdown string
    """
    lines = []
    for key, value in instructions.items():
        lines.append(f"## {key}")
        if isinstance(value, list):
            for item in value:
                lines.append(f"{item}")
        elif isinstance(value, str):
            lines.append(value)
        lines.append("")
    return "\n".join(lines)
