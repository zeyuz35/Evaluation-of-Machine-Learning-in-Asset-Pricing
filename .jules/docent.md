## 2024-03-25 - Replace cat with message for progress indicators

**Learning:** When using `cat()` for progress reporting (like printing dots in a loop or callback), it violates the style guideline of not using `cat()` in production code. Converting `cat()` to `message()` requires `appendLF = FALSE` to prevent `message()` from appending a newline after every character, which would break the visual effect of printing progress dots on a single line.

**Action:** Replace `cat(".")` with `message(".", appendLF = FALSE)` and `cat("\\n")` with `message("\\n", appendLF = FALSE)` for inline progress indicators.
