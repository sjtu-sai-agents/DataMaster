You are analyzing the output from a grading script that evaluated a submission on the **test set**.

## Grading Output:

```
{grade_output}
```

## Instructions

The user has specified the optimization direction for this task:
- **Optimization Direction**: {direction} ("minimize" = lower scores are better, "maximize" = higher scores are better)

Analyze the grading output and provide the following information:

1. **metric**: The test set metric value reported in the output (extract the number, null if not found)
2. **lower_is_better**: Always use the user-specified direction: {lower_is_better}
3. **is_bug**: Whether the grading process failed (true if there are errors, no metric found, or timeout)
4. **has_submission**: Whether a submission was evaluated (true if a score is found)
5. **summary**: A brief summary of the grading result

Return a JSON object containing the following keys:

- `metric` (number or null): The test set metric value extracted from the output
- `lower_is_better` (boolean): {lower_is_better} (fixed based on user specification)
- `is_bug` (boolean): Whether the grading failed
- `has_submission` (boolean): Whether a score was found
- `summary` (string): A brief summary

**Output only pure JSON**. Do not include additional text, explanations, or formatting.
