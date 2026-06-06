## 2024-06-06 - Replace cat() with message()
**Learning:** The project's style guide requires `cat()` calls used for progress reporting or informational messages to be replaced with `base::message()`. Explicit newline characters (`\n`) should be removed since `message()` automatically appends a newline.
**Action:** Used `sed` in a loop to systematically replace `cat(...)` calls with `message(...)` in R files, verifying that trailing `\n` characters are properly removed.
