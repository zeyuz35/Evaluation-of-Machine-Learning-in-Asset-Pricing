## 2024-05-16 - Replace raw cat() with message()
**Learning:** R demo scripts should use `message()` instead of `cat()` for logging informative strings to ensure structured logging, and trailing newlines `\n` should be removed because `message()` appends newlines automatically.
**Action:** Replaced `cat()` with `message()` in xgboost demo scripts, ensuring properly structured logging without double newlines.
