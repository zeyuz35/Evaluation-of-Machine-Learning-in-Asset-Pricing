## 2024-05-14 - Remove library calls in R directory
**Learning:** The project's `agents.md` prohibits `library()` or `require()` calls within the `R/` directory. Found multiple instances of `library("reticulate")` and `library("xgboost")`. Wait, there are many `library(...)` calls in `.Rmd` and `.R` files. For a small structural improvement without breaking changes, picking ONE explicit namespace fix in `R/AWS/sagemaker_demo.R` is safe.
Actually, if `sagemaker_demo.R` has `library(tidyverse)` and `library(stringr)` and `library("reticulate")`, removing just `library("reticulate")` and using explicit namespace for `reticulate::import` is one logical change.
Wait! I'll replace `library("reticulate")` with explicit `reticulate::import('sagemaker')`.

Actually, wait, I could instead modify `gpu_accelerated.R` to just explicitly use `xgboost::...`.
But let's look at `R/AWS/sagemaker_demo.R`
It uses `read_csv`, `filter`, `mutate`, `as.integer`, `select`, `sample_frac`, `anti_join`, `write_csv` (tidyverse functions).
It uses `str_split` (stringr).
If I remove `library("reticulate")`, I'll fix the `reticulate` namespace.
To fix *all* library calls in `R/AWS/sagemaker_demo.R`, I'd need to explicitly namespace all of them, which is slightly more error-prone, but more complete for that file.

Alternatively, `R/xgboost/demo/gpu_accelerated.R` uses `library('xgboost')`, and calls `xgb.DMatrix()`, `xgb.train()`. It's a very simple script, easy to fix. Let's fix `R/xgboost/demo/gpu_accelerated.R`.
