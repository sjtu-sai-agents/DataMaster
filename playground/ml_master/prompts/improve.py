"""Improve Agent Prompt

This prompt is used to improve existing solutions.
Fixed prompt content is loaded from external files in new_prompts/ directory.
"""


def get_introduction(obfuscate: bool = False) -> str:
    """Get the introduction for improve agent.

    Args:
        obfuscate: If True, use obfuscated (non-Kaggle) version

    Returns:
        Introduction string
    """
    if obfuscate:
        return (
            "You are an expert machine learning engineer attempting a task. You are provided with a previously developed "
            "solution below and should improve it in order to further increase the (test time) performance. "
            "For this you should first outline a brief plan in natural language for how the solution can be improved and "
            "then implement this improvement in Python based on the provided previous solution. "
        )
    return (
        "You are a Kaggle grandmaster attending a competition. You are provided with a previously developed "
        "solution below and should improve it in order to further increase the (test time) performance. "
        "For this you should first outline a brief plan in natural language for how the solution can be improved and "
        "then implement this improvement in Python based on the provided previous solution. "
    )


def get_prompt(
    task_desc: str,
    memory: str,
    data_preview: str,
    previous_code: str,
    execution_output: str,
    impl_guideline: list,
    obfuscate: bool = False,
) -> dict:
    """Build the complete improve prompt.

    Args:
        task_desc: Task description
        memory: Memory of previous solutions
        data_preview: Data preview string
        previous_code: Previous solution code
        execution_output: Execution output from previous solution
        impl_guideline: Implementation guidelines
        obfuscate: If True, use obfuscated version

    Returns:
        Dictionary with 'introduction' and 'user_prompt' keys
    """
    introduction = get_introduction(obfuscate)
    implementation_guideline_str = chr(10).join(f"- {g}" for g in impl_guideline)

    # load improvement guideline
    improvement_guideline_template_path = (
        "playground/ml_master/prompts/new_prompts/improvement_guideline.md"
    )
    data_enhancement_guideline_path = (
        "playground/ml_master/prompts/new_prompts/data_enhance_zh.md"
    )
    algo_enhancement_guideline_path = (
        "playground/ml_master/prompts/new_prompts/algo_enhance_zh.md"
    )
    with open(data_enhancement_guideline_path, "r", encoding="utf-8") as file:
        data_enhancement_prompt = file.read()
    with open(algo_enhancement_guideline_path, "r", encoding="utf-8") as file:
        algo_enhancement_prompt = file.read()
    with open(improvement_guideline_template_path, "r", encoding="utf-8") as file:
        improvement_guideline_template = file.read()

    improvement_guideline = improvement_guideline_template.format(
        data_enhancement_prompt=data_enhancement_prompt,
        algo_enhancement_prompt=algo_enhancement_prompt,
    )

    # load user prompt template
    user_prompt_template_path = (
        "playground/ml_master/prompts/new_prompts/improve_agent_prompts.md"
    )
    with open(user_prompt_template_path, "r", encoding="utf-8") as file:
        user_prompt_template = file.read()
        user_prompt = user_prompt_template.format(
            task_desc=task_desc,
            memory=memory,
            improvement_guideline=improvement_guideline,
            implementation_guideline_str=implementation_guideline_str,
            data_preview=data_preview,
            previous_code=previous_code,
            execution_output=execution_output,
        )

    return {
        "introduction": introduction,
        "user_prompt": user_prompt,
    }


if __name__ == "__main__":
    user_prompt = get_prompt(
        "NOT_FINISHED",
        "NOT_FINISHED",
        "NOT_FINISHED",
        "NOT_FINISHED",
        "NOT_FINISHED",
        ["NOT_FINISHED"],
    )["user_prompt"]

    print(user_prompt)
