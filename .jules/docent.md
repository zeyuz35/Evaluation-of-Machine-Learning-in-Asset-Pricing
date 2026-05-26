## 2024-05-18 - Convert cat to message
**Learning:** Found multiple instances where `cat()` was used for informational messages with explicit newlines `\n`, which violates the project's styling agents.
**Action:** Replaced `cat()` with `message()` and used `paste0()` to concatenate variables, removing explicit `\n` as `message()` appends newlines automatically.
