# Data Merge Methods

Use this reference when the black node needs to merge original competition data with local external datasets. The main objective is not just to concatenate data, but to do so in a way that is label-safe, validation-safe, and distribution-aware.

## Core Principle

Always treat the competition training set as the anchor dataset. External data is supplementary. It may improve coverage, balance, or robustness, but it must not redefine the validation protocol or overwhelm the task distribution blindly.

---

## Merge Checklist

Before merging any external source, verify all of the following:

- The data is already local and allowed to be used
- Format is understood and loadable
- Label mapping exists and is trustworthy
- External samples can be separated cleanly from validation
- Image or row schema is compatible with the competition task
- Source quality is acceptable

If any one of these is false, do not merge that subset yet.

---

## Recommended Merge Order

1. Load original competition train and validation first.
2. Freeze validation protocol:
   - if `input/val.csv` exists, use it
   - otherwise split only original competition train into train and val first
3. Load external datasets separately.
4. Apply source-specific filtering:
   - invalid samples
   - unsupported labels
   - ambiguous mappings
   - broken files
5. Apply label mapping.
6. Merge only into training fold.
7. Log counts by source and class.

---

## Image Merge Patterns

### Pattern A: Exact-label merge

Use when:

- External source labels map cleanly to competition labels
- Example: `Apple___healthy -> healthy`

Method:

- Convert external label strings using explicit mapping
- Append resulting samples to train set
- Preserve source metadata

This is the safest merge pattern.

---

### Pattern B: Partial-class merge

Use when:

- Only some external classes are trustworthy
- Example: healthy and scab map cleanly, but another class is semantically ambiguous

Method:

- Keep only the trustworthy mapped classes
- Drop the ambiguous remainder
- Do not force every external class to be used

This is often better than noisy full merge.

---

### Pattern C: Ratio-capped merge

Use when:

- External data volume is much larger than competition data
- External source distribution is very different from the competition

Method:

- Cap per-class or per-source external contribution
- Sample a bounded number of external examples
- Keep source diversity without letting one source dominate

Recommended when:

- external source is over 2x to 5x the original class count
- external labels are weaker than competition labels

---

### Pattern D: Minority-class support merge

Use when:

- Certain competition classes are underrepresented
- External source helps only a few weak classes

Method:

- Count competition train samples per class
- Add external data only for minority classes
- Keep majority classes mostly anchored in competition data

This is often a high-value, low-risk merge strategy.

---

## Path Rewriting Guidance

External image sources often store files under different root layouts. Normalize path handling before building the final dataset.

Recommended practice:

- Resolve source root once
- Build absolute or workspace-stable paths
- Store final sample tuples in a unified form

For image tasks, a stable sample record usually looks like:

```python
{
    "image_path": "/abs/path/to/file.jpg",
    "label": "healthy",
    "source": "plantvillage"
}
```

For parquet/image-bytes tasks, a stable sample record usually looks like:

```python
{
    "parquet_path": "/abs/path/to/train-00001.parquet",
    "row_idx": 123,
    "label": "frog_eye_leaf_spot",
    "source": "unified_plant_disease"
}
```

The goal is to make downstream dataset code treat all sources uniformly.

---

## Label Mapping Rules

Use explicit mappings only. Do not infer labels from folder names or free text unless there is a written rule.

Safe mapping examples:

- exact synonym
- dataset-specific canonical disease name that clearly matches competition label

Risky mapping examples:

- broad symptom name to narrow disease class
- one external class to multiple competition classes
- visually similar but medically or scientifically distinct categories

If mapping is risky:

- mark it as high-risk
- exclude it by default
- only include it if there is a task-specific justification

---

## Merge Logging Requirements

Every merge should produce enough information to debug downstream issues. At minimum log:

- original train size
- original val size
- external source sizes before filtering
- external source sizes after filtering
- per-class counts after merge
- dropped class names and reasons

This matters because later metric shifts are otherwise impossible to interpret.

---

## Anti-Patterns

Do not do these:

- Concatenate original and external data and then run `train_test_split` on the combined dataset
- Use external labels without checking mapping quality
- Mix sources without tracking provenance
- Let external source class distribution dominate silently
- Assume all image folders share the same structure

---

## Modality-Specific Notes

### Tabular

- Align schema before concatenation
- Recompute train-only imputers after merge, not before
- Confirm target column semantics match exactly

### Text

- Normalize text before deduplication
- Deduplicate competition and external train rows when necessary
- Be conservative with label mapping across sentiment, toxicity, or multi-class taxonomies

---

## Practical Recommendation

When the black node is under uncertainty, prefer this order of safety:

1. exact-label merge
2. partial-class merge
3. minority-class support merge
4. ratio-capped merge

Avoid full unrestricted merge unless source quality and label semantics are both strong.
