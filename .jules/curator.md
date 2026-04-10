## 2025-04-11 - Base as.matrix() drops time series attributes

**Learning:** Base R's `as.matrix()` strips class, time indices, and scaling metadata from time-series objects like `ts`, `xts`, and `zoo`. This can cause silent data integrity issues when time-series metadata is lost during matrix coercion.

**Action:** Use a custom function to coerce these objects safely while explicitly preserving their original class and non-structural attributes. The function `as_matrix_preserve()` was created in `R/utils/utils_integrity.R`.
