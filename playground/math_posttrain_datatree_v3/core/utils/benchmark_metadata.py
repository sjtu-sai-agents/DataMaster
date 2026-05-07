"""Benchmark metadata for agent context."""

BENCHMARK_METADATA = {
    "aime_2025": {
        "type": "Math Competition",
        "description": "AIME (American Invitational Mathematics Examination) 2025 problems. High-difficulty competition math requiring multi-step reasoning.",
        "input_format": "Problem statement (text)",
        "output_format": 'Step-by-step solution; last line must be "ANSWER: <integer 0-999>"',
        "example": "Problem: Find the number of positive integers n ≤ 1000 such that n² + n + 1 is divisible by 7.\\nAnswer: ...\\nANSWER: 143"
    },
    "gsm8k": {
        "type": "Grade School Math",
        "description": "Grade school math word problems requiring arithmetic and basic reasoning.",
        "input_format": "Word problem (text)",
        "output_format": 'Step-by-step solution; last line must be "ANSWER: <number>"',
        "example": "Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\\nAnswer: 16 - 3 - 4 = 9 eggs. She makes 9 * $2 = $18 every day.\\nANSWER: 18"
    },
    "human_eval": {
        "type": "Code Generation",
        "description": "Python function completion from docstrings. Evaluate functional correctness via test cases.",
        "input_format": "Function signature + docstring",
        "output_format": "Complete Python function implementation",
        "example": 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\\n    False\\n    """\\n    # Implementation here'
    },
    "healthbench_easy": {
        "type": "Medical QA",
        "description": "Medical question answering with safety, accuracy, and appropriate hedging. Multi-turn conversations.",
        "input_format": "Multi-turn conversation (user questions + assistant responses)",
        "output_format": "Medical advice with appropriate caveats and recommendations",
        "example": "User: I have a headache. Should I take aspirin?\nAssistant: Aspirin can help with headaches for many people. However, if you have certain conditions or take specific medications, aspirin might not be suitable. It's best to consult with a healthcare provider if you're unsure."
    },
    "bfcl": {
        "type": "Function Calling",
        "description": "Berkeley Function Calling Leaderboard. Evaluate ability to call functions/APIs correctly.",
        "input_format": "User query + available function definitions",
        "output_format": "Function call with correct parameters in JSON format",
        "example": "User: What's the weather in San Francisco?\\nFunctions: get_weather(location: str, unit: str)\\nOutput: {{\"name\": \"get_weather\", \"arguments\": {{\"location\": \"San Francisco\", \"unit\": \"celsius\"}}}}"
    },
    "gpqa_main": {
        "type": "Graduate-level Science QA",
        "description": "Graduate-level science questions (physics, chemistry, biology) requiring expert knowledge.",
        "input_format": "Multiple-choice question with 4 options",
        "output_format": 'Reasoning allowed; last line must be "ANSWER: <A|B|C|D>"',
        "example": "Question: What is the primary mechanism of action for statins?\nA) Inhibit HMG-CoA reductase\nB) Block calcium channels\nC) Inhibit ACE\nD) Block beta receptors\nAnswer: A"
    },
    "arena_hard_writing": {
        "type": "Creative Writing",
        "description": "Arena Hard writing tasks. Generate high-quality creative or technical writing.",
        "input_format": "Writing prompt or instruction",
        "output_format": "Extended text response (essay, story, explanation, etc.)",
        "example": "Prompt: Write a short story about a time traveler who accidentally changes history.\nOutput: [Extended creative writing response]"
    }
}


def get_benchmark_info(benchmark_id: str) -> str:
    """Format benchmark metadata for agent prompts."""
    meta = BENCHMARK_METADATA.get(benchmark_id)
    if not meta:
        return f"Benchmark: {benchmark_id} (no metadata available)"

    return f"""## Target Benchmark: {benchmark_id}

**Type**: {meta['type']}

**Description**: {meta['description']}

**Input Format**: {meta['input_format']}

**Output Format**: {meta['output_format']}

**Example**:
```
{meta['example']}
```"""
