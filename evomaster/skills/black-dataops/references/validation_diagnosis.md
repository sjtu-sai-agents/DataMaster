# Validation Diagnosis

Use this reference when validation predictions are available and the goal is to infer whether the data itself is the bottleneck. This is not for changing the validation protocol. It is for diagnosing data problems from validation behavior.

## Purpose

A black node often sees one of these situations:

- training loss decreases but validation metric stalls
- a few classes have near-zero recall
- one class is overpredicted
- thresholds drift far away from the default
- external data helps some classes but hurts overall stability

These are often signs of data quality or merge issues rather than model-capacity issues.

---

## What To Inspect

When validation predictions are available, inspect:

- per-class precision, recall, and F1
- confusion tendencies or one-vs-rest failure patterns
- threshold chosen per class
- samples with highest-confidence wrong predictions
- classes with very low support
- source-specific failures if provenance is available

---

## Common Diagnosis Patterns

### Pattern 1: One class is never predicted

Possible causes:

- too few training examples
- label mapping dropped that class from external data
- augmentations destroy the signal for that class
- the class is merged into a neighboring label by mistake

Recommended actions:

- inspect class counts
- inspect whether external samples for that class were skipped
- add minority-class support or sampler logic

---

### Pattern 2: One class is predicted too often

Possible causes:

- class imbalance
- noisy label mapping from external data into that class
- train distribution shifted after merge
- threshold too low for that class

Recommended actions:

- compare class counts before and after merge
- inspect high-confidence false positives
- raise suspicion on broad external labels mapped into this class

---

### Pattern 3: Optimized thresholds become extreme

Possible causes:

- class calibration drift due to noisy external data
- class prior mismatch between train and validation
- low-quality positive samples

Interpretation guide:

- very high threshold may indicate many false positives
- very low threshold may indicate the model rarely emits strong logits for that class

Recommended actions:

- review label quality for that class
- review source balance and augmentation severity

---

### Pattern 4: Metric improves for seen classes but worsens globally

Possible causes:

- external merge helps common classes but hurts rare ones
- macro metric is exposing weak minority-class behavior
- validation remains anchored to competition domain while external data is off-domain

Recommended actions:

- compare per-class deltas, not just overall metric
- consider partial-class or ratio-capped merge
- reduce influence of noisy external subsets

---

### Pattern 5: High training performance, unstable validation performance

Possible causes:

- overfitting
- bad augmentations
- duplicate or low-diversity external samples
- train/val mismatch

Data-oriented checks:

- verify external images are not near duplicates of each other in a narrow way
- check image quality variability
- reduce augmentation aggressiveness if labels are already noisy

---

## How Validation Helps Find Data Problems

Validation predictions are useful for identifying:

- missing classes in merged training data
- overaggressive label mapping
- source-specific noise
- thresholds compensating for bad data instead of good learning
- classes whose visual or semantic definition is under-specified

Validation predictions are not enough to prove:

- that a risky label mapping is scientifically correct
- that private leaderboard performance will improve
- that a distribution shift is solved just because one fold improved

---

## Recommended Debugging Outputs

If the code can print diagnostics, prefer:

- per-class support in train and val
- per-class F1
- per-class threshold if threshold tuning is used
- list of classes with support below threshold
- count of external samples merged per class
- count of dropped external samples per class or reason

These diagnostics make it much easier to decide whether the next change should be:

- more cleaning
- less noisy merge
- better class balancing
- milder augmentation

---

## Warning Signs That Suggest Data Issues

- Thresholds differ wildly across classes
- Classes with many external samples still have near-zero recall
- External merge improves train metric but harms val metric
- Validation metric becomes brittle across nearby runs
- One source dominates false positives

When these appear, fix data first before making model changes.

---

## Validation Safety Rules

- Never redefine validation just to get a cleaner metric.
- Never move external data into validation to make tuning easier.
- If validation is fixed by `input/val.csv`, preserve it.
- If the current branch already uses the fixed validation correctly, keep that logic stable.

The value of validation diagnosis comes from comparability. Once that is broken, the diagnosis is unreliable.
