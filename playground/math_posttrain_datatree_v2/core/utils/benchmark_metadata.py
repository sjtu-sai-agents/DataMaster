"""Benchmark metadata for agent context."""

BENCHMARK_METADATA = {
    "aime_2025": {
        "type": "Math Competition",
        "description": "AIME (American Invitational Mathematics Examination) 2025 problems. High-difficulty competition math requiring multi-step reasoning.",
        "input_format": "Problem statement (text)",
        "output_format": 'Step-by-step solution; last line must be "ANSWER: <integer 0-999>"',
        "example": "Problem: Find the number of positive integers n ≤ 1000 such that n² + n + 1 is divisible by 7.\\nAnswer: ...\\nANSWER: 143",
        "data_preparation_guide": """
**Answer Extraction and Data Preparation:**

AIME benchmark requires final answers to be integers in range [0, 999], but training data can be more flexible.

1. **Training data sources can include:**
   - AIME-specific problems (ideal, but limited availability)
   - Competition math problems with various answer formats
   - Math problems with fractional, negative, or large answers (focus on reasoning quality)
   - Problems from AMC, USAMO, IMO, etc. (even if answer format differs)
   - **Key criterion: Strong multi-step mathematical reasoning, not just answer format**

2. **Answer extraction from source data:**
   - `\\boxed{143}` or `\\boxed{n=143}` → extract 143
   - `n = 143` or `The answer is 143` → extract 143
   - `\\frac{13}{6}` → can keep if reasoning is valuable; model will learn to output integers
   - `\\textbf{(A)} 26` → extract 26 (ignore multiple choice label)
   - Plain integers: `143`, `0`, `999` → use directly
   - **For non-integer answers**: Keep the problem if reasoning is strong; standardize output to closest integer or skip only if answer is completely incompatible

3. **Output format standardization:**
   - Preserve the step-by-step reasoning from source data (this is most important)
   - Replace the final answer line with: `ANSWER: <integer>`
   - If source answer is not an integer, either:
     - Round/convert to integer if reasonable (e.g., 26.0 → 26)
     - Or skip this specific example (but don't discard the entire dataset)
   - Ensure the last non-empty line is exactly `ANSWER: <integer>` where integer is 0-999

4. **Data volume strategy:**
   - Start small: 500-1000 examples for initial validation
   - If results are promising, scale up to 2000-5000
   - Monitor for overfitting; more data isn't always better
   - Prefer high-quality reasoning over large quantity

5. **Quality checks:**
   - Verify reasoning steps are present and coherent (not just answer-only)
   - Check that final output format is `ANSWER: <integer>` where integer is 0-999
   - Remove exact duplicates across sources
   - Ensure diversity in problem types and difficulty levels
"""
    },
    "gsm8k": {
        "type": "Grade School Math",
        "description": "Grade school math word problems requiring arithmetic and basic reasoning.",
        "input_format": "Word problem (text)",
        "output_format": 'Step-by-step solution; last line must be "ANSWER: <number>"',
        "example": "Problem: Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning and bakes muffins for her friends every day with four. She sells the remainder at the farmers' market daily for $2 per fresh duck egg. How much in dollars does she make every day at the farmers' market?\\nAnswer: 16 - 3 - 4 = 9 eggs. She makes 9 * $2 = $18 every day.\\nANSWER: 18",
        "data_preparation_guide": """
**Answer Extraction from Source Data:**

1. **Common formats:** `\\boxed{18}`, `$18`, `18 dollars`, `The answer is 18`
2. **Extract numeric value:** Remove currency symbols, units, and text wrappers
3. **Output format:** `ANSWER: <number>` (can be integer or decimal)
4. **Keep reasoning:** Preserve step-by-step arithmetic from source
"""
    },
    "human_eval": {
        "type": "Code Generation",
        "description": "Python function completion from docstrings. Evaluate functional correctness via test cases.",
        "input_format": "Function signature + docstring",
        "output_format": "Complete Python function implementation",
        "example": 'def has_close_elements(numbers: List[float], threshold: float) -> bool:\\n    """ Check if in given list of numbers, are any two numbers closer to each other than given threshold.\\n    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)\\n    False\\n    """\\n    # Implementation here',
        "data_preparation_guide": """
**Data Preparation:**

1. **Input format:** Function signature + docstring (may include examples)
2. **Output format:** Complete, syntactically correct Python function
3. **Quality:** Ensure code is properly indented and follows Python conventions
4. **No answer extraction needed:** Code completion task, not QA
"""
    },
    "healthbench_easy": {
        "type": "Medical QA",
        "description": "Medical question answering with safety, accuracy, and appropriate hedging. Multi-turn conversations.",
        "input_format": "Multi-turn conversation (user questions + assistant responses)",
        "output_format": "Medical advice with appropriate caveats and recommendations",
        "example": "User: I have a headache. Should I take aspirin?\nAssistant: Aspirin can help with headaches for many people. However, if you have certain conditions or take specific medications, aspirin might not be suitable. It's best to consult with a healthcare provider if you're unsure.",
        "data_preparation_guide": """
**Data Preparation:**

1. **Safety first:** Ensure responses include appropriate medical disclaimers
2. **Hedging:** Responses should acknowledge uncertainty and recommend professional consultation
3. **Multi-turn:** Preserve conversation context across turns
4. **No answer extraction needed:** Open-ended medical advice task
"""
    },
    "bfcl": {
        "type": "Function Calling",
        "description": "Berkeley Function Calling Leaderboard. Evaluate ability to call functions/APIs correctly.",
        "input_format": "User query + available function definitions",
        "output_format": "Function call with correct parameters in JSON format",
        "example": "User: What's the weather in San Francisco?\\nFunctions: get_weather(location: str, unit: str)\\nOutput: {{\"name\": \"get_weather\", \"arguments\": {{\"location\": \"San Francisco\", \"unit\": \"celsius\"}}}}",
        "data_preparation_guide": """
**Data Preparation:**

1. **Input:** User query + function schema/definitions
2. **Output:** Valid JSON function call with correct parameter mapping
3. **Validation:** Ensure JSON is well-formed and parameters match schema
4. **No answer extraction needed:** Structured output task
"""
    },
    "gpqa_main": {
        "type": "Graduate-level Science QA",
        "description": "Graduate-level science questions (physics, chemistry, biology) requiring expert knowledge.",
        "input_format": "Multiple-choice question with 4 options",
        "output_format": 'Reasoning allowed; last line must be "ANSWER: <A|B|C|D>"',
        "example": "Question: What is the primary mechanism of action for statins?\nA) Inhibit HMG-CoA reductase\nB) Block calcium channels\nC) Inhibit ACE\nD) Block beta receptors\nAnswer: A",
        "data_preparation_guide": """
**Answer Extraction from Source Data:**

1. **Common formats:** `\\boxed{A}`, `The answer is A`, `(A)`, `Option A`
2. **Extract letter:** Must be exactly one of A, B, C, or D
3. **Output format:** `ANSWER: <A|B|C|D>` (single letter, uppercase)
4. **Keep reasoning:** Preserve scientific explanation from source
5. **Skip if:** Answer is not clearly A/B/C/D or multiple answers given
"""
    },
    "arena_hard_writing": {
        "type": "Creative Writing",
        "description": "Arena Hard writing tasks. Generate high-quality creative or technical writing.",
        "input_format": "Writing prompt or instruction",
        "output_format": "Extended text response (essay, story, explanation, etc.)",
        "example": "Prompt: Write a short story about a time traveler who accidentally changes history.\nOutput: [Extended creative writing response]",
        "data_preparation_guide": """
**Data Preparation:**

1. **Input:** Clear writing prompt or instruction
2. **Output:** High-quality extended text (300+ words typical)
3. **Quality:** Ensure coherent, well-structured, engaging writing
4. **No answer extraction needed:** Open-ended creative task
"""
    }
}


def get_benchmark_info(benchmark_id: str) -> str:
    """Format benchmark metadata for agent prompts."""
    meta = BENCHMARK_METADATA.get(benchmark_id)
    if not meta:
        return f"Benchmark: {benchmark_id} (no metadata available)"

    sections = [
        f"## Target Benchmark: {benchmark_id}",
        f"**Type**: {meta['type']}",
        f"**Description**: {meta['description']}",
        f"**Input Format**: {meta['input_format']}",
        f"**Output Format**: {meta['output_format']}",
        f"**Example**:\n```\n{meta['example']}\n```"
    ]

    # Add data preparation guide if available
    if meta.get('data_preparation_guide'):
        sections.append(meta['data_preparation_guide'])

    return "\n\n".join(sections)
