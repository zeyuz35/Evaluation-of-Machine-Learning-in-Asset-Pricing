## 2024-05-30 - Preserving attributes on as.matrix() coercion

**Learning:** Base R's `as.matrix()` inherently drops critical non-structural attributes such as indices (`index`, `tsp`) and scaling parameters (`scaled:center`, `scaled:scale`). When converting time-series objects to computationally safe matrices (dropping the class so method dispatch is generic), we must manually save and re-attach these attributes so downstream code (e.g. backtransformation) does not fail silently.

**Action:** Created `as_matrix_preserve()` which safely captures non-structural attributes, calls `as.matrix()`, and reinstates attributes while omitting the `class` attribute. Applied this function to `.R` scripts like `R/AWS/sagemaker_run.R` where matrices are extracted from time series structures for deep learning contexts but still need their metadata attached.
