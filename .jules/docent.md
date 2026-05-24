## 2024-05-18 - convert raw cat() to message()
**Learning:** Raw output with `cat()` should be formatted into structured `message()` in R.
**Action:** Replace `cat()` calls with `message()` ensuring multiple string arguments are passed via `paste0()` and formatting explicitly avoids trailing `\n`.
