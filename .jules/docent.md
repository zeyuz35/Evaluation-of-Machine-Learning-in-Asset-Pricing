## 2024-05-30 - Replace cat() with message() in scripts
**Learning:** The project's `agents.md` file specifies under "Content and Communication Standards" (line 39): "- Replace `cat()` calls used for progress reporting or informational messages with `base::message()`".
**Action:** Replace `cat()` with `message()` in demo R scripts like `R/xgboost/demo/generalized_linear_model.R` or `R/xgboost/demo/cross_validation.R`. Since `message()` automatically appends a newline, any trailing `\n` in the string passed to `cat()` should be removed.
