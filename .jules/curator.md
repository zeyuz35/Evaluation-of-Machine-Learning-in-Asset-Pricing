## 2024-05-15 - Matrix Coercion Attribute Loss

**Learning:** Base R's `as.matrix()` strips non-structural attributes such as `scaled:center`, `scaled:scale`, or any custom user-attached attributes (`custom_attr`) from objects (like those returned by `scale()` or specific `ts` attributes). This leads to silent metadata loss which breaks downstream inverse transformations (like backtransforming scaled predictions).

**Action:** Created and enforce usage of `as_matrix_preserve()` helper in `R/utils_integrity.R` which securely captures all original attributes before coercion and merges back non-structural attributes after coercion to guarantee no scaling or transformation data is silently dropped.
