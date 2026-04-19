## 2024-04-19 - Class and Attribute Preservation in Time Series

**Learning:** Base R `as.matrix()` silently drops structural (like `index` in `xts`) and non-structural (`scaled:scale`, etc.) attributes from time-series objects.

**Action:** Implement `as_matrix_preserve()` wrapper inside `R/utils_integrity.R` to safely coerce `ts`, `xts`, and `zoo` objects to a matrix format without losing essential scaling/index attributes, preserving the original object's class to maintain consistency in downstream predictions.
