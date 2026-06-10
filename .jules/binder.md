## 2024-05-24 - Remove library() calls and use explicit namespacing
**Learning:** Found scripts with missing dependencies that throw errors when executed because packages like `umap` are missing. Also found a lot of `library()` calls, which violate project guidelines.
**Action:** Replace `library()` calls with explicit namespacing (e.g. `dplyr::filter`) in R scripts.
