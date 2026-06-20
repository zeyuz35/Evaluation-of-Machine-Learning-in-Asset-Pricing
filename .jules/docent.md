## 2024-06-20 - [Progress Reporting Standardization]
**Learning:** Found multiple `cat()` calls inside `callback_lambda` specifically used for progress reporting (e.g., `cat("\n")` and `cat(".")` inside loops) in various Rmd files, which goes against the `agents.md` rule.
**Action:** Replaced `cat("\n")` with `message("")` and `cat(".")` with `message(".", appendLF = FALSE)`.
