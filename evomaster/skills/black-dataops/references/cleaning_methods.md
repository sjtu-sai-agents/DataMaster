# Cleaning Methods

Use this reference when the black node needs to improve data quality before training. The goal is to make data loading stable, interpretable, and reusable. Prefer explicit filtering and normalization over implicit assumptions.

## How To Choose

- If the task is image classification or detection, start with the image methods below.
- If the task is tabular, start with missing values, feature types, and categorical cleanup.
- If the task is text, start with normalization, deduplication, and label mapping.
- If you are unsure, first inspect a sample of rows or files and classify the task modality.

---

## Image Methods

### `image_validity_filter`

Use when:

- Some files may be zero-byte, truncated, corrupted, or non-image files with image extensions
- Training fails with PIL decode errors
- A downloaded dataset contains hidden files or metadata files

What it should do:

- Skip zero-byte files
- Attempt lightweight decode validation with `PIL.Image.open(...).verify()`
- Re-open successfully verified images in normal load mode before training
- Record how many samples were removed and from which source

Recommended checks:

- Extension is in an allowed set such as `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`
- File size is greater than zero
- PIL decode succeeds
- Optional: minimum width and height threshold

Implementation notes:

- Use this before constructing the final sample list
- Keep the filter deterministic
- Never let a single bad file crash the whole DataLoader

Common failure modes prevented:

- `PIL.UnidentifiedImageError`
- broken batch workers from malformed files
- noisy external datasets silently poisoning training

---

### `image_size_normalizer`

Use when:

- External images have variable sizes
- Different sources use grayscale, RGBA, CMYK, or mixed color modes
- DataLoader fails because tensors have different shapes

What it should do:

- Convert every image to a known color mode, usually `RGB`
- Ensure transforms always produce the same tensor shape
- Optionally resize early to reduce extreme aspect-ratio problems

Recommended policy:

- Convert all inputs with `image.convert("RGB")`
- Use one safe training transform and one safe eval transform
- For fragile mixed-source datasets, resize before stochastic crop

Good defaults for classification:

- train: `Resize -> RandomCrop/RandomResizedCrop -> Flip -> ColorJitter -> ToTensor -> Normalize`
- eval: `Resize -> CenterCrop or exact Resize -> ToTensor -> Normalize`

Important caution:

- Do not let external images keep their raw original shapes if your collate path expects fixed tensors
- If source images are highly inconsistent, prefer a safer transform over an aggressive augmentation recipe

---

### `image_label_quality_check`

Use when:

- External classes are mapped manually
- Some classes look underrepresented or suspicious
- The model overpredicts one class or collapses on rare classes

What it should inspect:

- Class counts before and after external merge
- Very small classes
- Labels with unusually high loss or poor validation precision/recall
- Mapped classes whose semantics are weak or ambiguous

Recommended outputs:

- Per-class sample counts
- Classes below a configurable minimum count
- Warning list of high-risk mappings such as approximate disease synonym mapping

Heuristics:

- Mark classes as small if they are less than 1 percent of training set or below an absolute threshold
- Mark mappings as risky if the external source label is broader, narrower, or semantically adjacent rather than exact
- If the external label cannot be mapped with confidence, drop it instead of forcing it

---

### `image_augmentation_pack`

Use when:

- The model is overfitting
- External data introduces domain shift
- The task is image classification and only DataLoader-layer changes are allowed

What belongs here:

- Resize and crop policy
- Horizontal or vertical flips when semantically valid
- Mild color jitter
- Small rotation if class semantics are orientation-invariant
- Optional source-specific normalization hooks

What usually does not belong here:

- MixUp or CutMix if the task is strictly framed as data cleaning only
- Heavy task-specific augmentation that changes semantic content
- Training-loop-only logic unless the project explicitly treats it as part of the data recipe

Recommended style:

- Keep augmentations moderate and composable
- Prefer one stable training transform over many fragile variants
- Use stronger augmentation only after data validity and label quality are already stable

---

### `image_external_merge`

Use when:

- Competition images are being combined with local external image sources
- Different sources use different label strings or directory layouts
- The black node must merge data without changing the model architecture

What it should do:

- Load original samples and external samples separately first
- Apply label mapping before concatenation
- Drop unmappable or low-confidence external samples
- Keep provenance metadata such as `source=competition`, `source=plantvillage`, `source=kashmiri`

Required safeguards:

- External samples must enter training only, not validation
- Log the number of included and skipped samples per source
- Preserve original competition labels as the authority

Good merge strategies:

- Full merge only when label semantics are strong and source quality is acceptable
- Partial merge for selected classes only
- Ratio-capped merge to prevent external data from overwhelming the original competition distribution

---

## Tabular Methods

### `tabular_missing_value_handler`

Use when:

- Columns contain NaN, empty strings, or sentinel values like `-999`

What it should do:

- Normalize missing markers to one convention
- Impute numeric and categorical columns explicitly
- Add optional missingness indicators for important columns

Preferred policy:

- Numeric: median or domain-safe constant
- Categorical: explicit `"__missing__"` token
- Never rely on implicit pandas casting to hide missing data issues

---

### `tabular_categorical_cleaner`

Use when:

- Categories have inconsistent casing, whitespace, punctuation, or aliases

What it should do:

- Strip whitespace
- Normalize case only if semantics allow it
- Collapse known aliases
- Replace rare noisy tokens with an explicit fallback bucket when justified

---

### `tabular_outlier_clip`

Use when:

- Numeric features contain extreme values that destabilize training

What it should do:

- Clip or winsorize using explicit thresholds
- Apply thresholds derived from train-only statistics
- Log which columns were clipped

Important caution:

- Do not compute clip thresholds using validation or test data

---

### `tabular_feature_type_repair`

Use when:

- Numeric columns are loaded as strings
- Boolean columns use inconsistent encodings
- Dates are stored as free-form text

What it should do:

- Cast columns explicitly
- Parse dates into stable components
- Normalize booleans to a single representation
- Fail loudly on irreparable columns instead of silently coercing to junk

---

## Text Methods

### `text_normalizer`

Use when:

- Text sources have inconsistent spacing, casing, HTML artifacts, or encoding issues

What it should do:

- Normalize whitespace
- Remove obvious HTML or markup noise if not semantically meaningful
- Normalize Unicode where needed
- Preserve task-relevant symbols when they matter

---

### `text_deduplicator`

Use when:

- External text data may duplicate competition rows
- Near-identical templated text appears many times

What it should do:

- Exact dedup on normalized text
- Optional key-based dedup on `id`, `title + body`, or other stable fields
- Keep train and validation dedup checks separate

Important caution:

- Never deduplicate across train and validation in a way that changes the fixed validation protocol unless the benchmark explicitly allows it

---

### `text_label_mapper`

Use when:

- External text datasets use different label names or label schemas

What it should do:

- Map exact equivalents first
- Refuse many-to-many fuzzy mappings unless there is a documented policy
- Log dropped labels and coverage

---

## Decision Rules

- Start with validity and schema repair before augmentation.
- If external labels are uncertain, prioritize precision over recall when deciding what to include.
- If a change improves stability but not metric immediately, keep it if it prevents known failure modes.
- If a cleaning rule changes class balance heavily, review merge strategy before proceeding.
