## 2024-05-24 - Base R as.matrix() attribute stripping behavior

**Learning:** Base R's `as.matrix()` strips attributes such as scaling metadata (`scaled:center`, `scaled:scale`), `index`, and `tsp` when coercing complex objects like `ts` and `xts` into a matrix format. This breaks inversion functionality (e.g., reversing the scaling operation after a prediction).

**Action:** Created `as_matrix_preserve()` helper function in `R/utils_integrity.R` to safely coerce data objects to matrix format while explicitly preserving non-structural data and transformation attributes.
