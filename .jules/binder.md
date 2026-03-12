## 2025-01-20 - Rtsne duplication sensitivity

**Learning:** `Rtsne()` throws errors if duplicate rows are present in the input data frame, leading to failures in random sampling scenarios without deduplication.

**Action:** Wrap data frames passed to `Rtsne()` in `unique()` or ensure distinct rows prior to calling to prevent runtime duplication errors.
