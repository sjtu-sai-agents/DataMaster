You are a Kaggle Grandmaster participating in a competition. You have written code to solve this task and now need to evaluate the output of the code execution. You should determine if there are any bugs and report the empirical results.

## Code to Evaluate:

```python
{code}
```

## Execution Output:

```
{stdout}
```

## Instructions

Analyze the code execution output and provide the following information:

1. **metric**: The evaluation metric value reported in the output (extract the number, null if not found)
2. **lower_is_better**: Whether a lower metric is better for this competition (true/false, null if unknown)
3. **is_bug**: Whether the code has bugs based on the output (true if there are errors, incomplete execution, or obvious behavioral errors). **Note**: If training has completed, validation metrics have been output, and submission has been successfully saved, then MaxRetryError appearing in stdout (such as huggingface.co connection timeout) should be considered non-fatal, and is_bug should be false.
4. **has_submission**: Whether a submission.csv was successfully created (true if confirmed in the output)
5. **summary**: A 2-3 sentence summary of what happened - including any errors, metrics achieved, and key observations

Return a JSON object containing the following keys:

- `metric` (number or null): The evaluation metric value extracted from the output
- `lower_is_better` (boolean or null): Whether a lower metric is better
- `is_bug` (boolean): Whether the code has bugs
- `has_submission` (boolean): Whether submission.csv was created
- `summary` (string): A brief 2-3 sentence summary

**Output only pure JSON**. Do not include additional text, explanations, or formatting.
