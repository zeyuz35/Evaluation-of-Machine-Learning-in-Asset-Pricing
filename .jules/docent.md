## 2024-05-11 - Converting cat to message
**Learning:** Found several scripts using `cat()` for progress reporting dots which violates the `agent: Console Output and Messaging` guideline. R's `message()` is preferred over `cat()` for informational output. For printing characters without appending a new line, use `message(..., appendLF = FALSE)`.
**Action:** Replaced instances of `cat("\n")` with `message("")` and `cat(".")` with `message(".", appendLF = FALSE)`.
