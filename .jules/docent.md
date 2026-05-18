## 2024-05-18 - Convert cat() to message()
**Learning:** Raw output with `cat()` for progress and informational logging should be converted to `message()` or `warning()`, dropping trailing newlines as `message()` appends them automatically.
**Action:** Replace `cat` outputs in informational settings with `message` outputs and combine variables using `paste0` or `paste` when replacing `cat`'s multi-argument printing.
