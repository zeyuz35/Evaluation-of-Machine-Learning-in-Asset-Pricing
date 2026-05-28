## 2024-05-28 - Convert cat to message
**Learning:** `cat()` calls in R should be converted to `message()` for structured logging. Trailing `\n` should be removed because `message()` appends them automatically. Multiple arguments should be wrapped in `paste0()`.
**Action:** Always replace `cat()` with `message()` in R scripts and remove explicit newlines. Use `paste0` to combine variables and strings.
