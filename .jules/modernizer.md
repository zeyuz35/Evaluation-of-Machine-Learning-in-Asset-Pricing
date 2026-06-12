## 2024-06-12 - Update Legacy Iterators

**Learning:** The codebase heavily uses legacy `1:length(x)` and `1:nrow(df)` loop patterns, which can cause out-of-bounds errors if the vector or dataframe is empty (the `1:0` bug). They are also slower and less idiomatic than `seq_along(x)` and `seq_len(nrow(df))`.
**Action:** Replaced `1:length(x)` with `seq_along(x)` and `1:nrow(df)` with `seq_len(nrow(df))` across the R codebase.
