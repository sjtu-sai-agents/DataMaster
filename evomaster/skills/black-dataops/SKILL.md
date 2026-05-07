---
name: black-dataops
description: Data loading, cleaning, external-data merge, and validation-diagnosis guidance for black-node style ML agents. Use when a coding agent must improve data quality or integrate external datasets without changing the core model logic, especially for image tasks but also for tabular and text tasks.
license: Internal project usage
---

# Black DataOps Guide

This skill is for black-node style agents whose job is to improve data quality and data loading logic, not to search for data online and not to redesign the whole training pipeline.

Use this skill when the task requires any of the following:

- Clean broken or inconsistent inputs before training
- Integrate external datasets that are already available locally
- Align external labels to competition labels
- Merge original and external data without contaminating validation
- Inspect validation predictions to discover data-quality problems

## Scope

This skill is intentionally narrow.

Do:

- Modify `MyDataLoader.setup()` and closely related data-loading helpers
- Add safe preprocessing, filtering, and merge logic
- Keep validation logic comparable across nodes

Do not:

- Search or download new datasets
- Rewrite the whole model or optimizer stack unless the task explicitly asks for it
- Change validation protocol in a way that breaks comparability

## Recommended Workflow

1. Read the manifest or local dataset description first.
2. Identify dataset type:
   - image folder
   - parquet/image-bytes
   - tabular csv/parquet
   - text csv/json/parquet
3. Load the relevant reference file:
   - Cleaning methods: `references/cleaning_methods.md`
   - External merge rules: `references/data_merge_methods.md`
   - Validation-based diagnosis: `references/validation_diagnosis.md`
4. Apply the smallest safe change that improves data quality.
5. Preserve the validation split and keep external data out of validation unless the task explicitly provides a trusted external validation protocol.

## Reference Files

- `references/cleaning_methods.md`
  Read when the task is about data cleaning, normalization, augmentation, or label-quality checks.
- `references/data_merge_methods.md`
  Read when combining original competition data with external datasets, especially image-folder or parquet-image sources.
- `references/validation_diagnosis.md`
  Read when validation predictions, confusion patterns, or threshold behavior are being used to diagnose data problems.

## Practical Rules

- Prefer deterministic cleaning over clever but fragile heuristics.
- If label mapping is uncertain, skip the ambiguous external subset instead of forcing it in.
- If external data format is known via manifest or verified snippet, trust that source over guessed parsing logic.
- For image tasks, prioritize stability first:
  - valid decodable images
  - consistent color mode
  - consistent tensor shape
  - safe label mapping
- For tabular and text tasks, prefer explicit schema repair and filtering over implicit pandas coercion.

## Output Pattern

When using this skill inside a code-writing agent, the expected outcome is usually:

- A safer `MyDataLoader.setup()` implementation
- Short helper functions for loading and filtering data
- Clear logging about what was included, skipped, or remapped
- No silent leakage into validation
