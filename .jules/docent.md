## 2024-05-02 - Message instead of Cat for Callbacks

**Learning:** When printing dots `.` in `keras` lambda callbacks using `cat()`, `R CMD check` and style guidelines prefer `message()`. `message(".", appendLF = FALSE)` replicates `cat(".")` functionality while maintaining style compliance.

**Action:** Replace `cat("\n")` with `message("")` and `cat(".")` with `message(".", appendLF = FALSE)` in callback functions to maintain correct progress printing without newlines.
