## 2024-05-24 - Replace print/cat with message in R
**Learning:** Found instances of `cat` and `print` being used for structured logging in R scripts (like `R/xgboost/demo/cross_validation.R`).
**Action:** Replace them with `message` which is proper for structured logging and removes the need for explicit newline `\n`.
