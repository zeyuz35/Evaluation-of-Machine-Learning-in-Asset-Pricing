## 2024-05-24 - Replace cat with message in progress callbacks
**Learning:** Found multiple usages of `cat("\n")` and `cat(".")` in neural network progress bar callbacks, which violates the rule to use `message()` instead of `cat()`.
**Action:** Replace `cat("\n")` with `message()` and `cat(".")` with `message(".", appendLF = FALSE)`.
