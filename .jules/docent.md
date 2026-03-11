## 2024-05-18 - Replacing cat() with message() without newlines
**Learning:** `message()` automatically appends newlines, so when replacing `cat(".")` to print dots on a single line for progress indicators, we cannot simply use `message(".")` because it will print each dot on a new line. We can use `message(".", appendLF = FALSE)`.
**Action:** Always use `appendLF = FALSE` when replicating the behavior of `cat()` for progress dots using `message()`.
