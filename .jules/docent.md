## 2024-06-17 - Replace print/cat with structured logging/message

**Learning:** `print()` and `cat()` are widely used in demo files inside `R/xgboost/demo/` for console output, which is not aligned with professional presentation and the instructions say "Enforce clear messaging and logging (e.g., replacing raw print statements with proper logging/messages)."
**Action:** Replace `print(...)` and `cat(...)` in `R/xgboost/demo/basic_walkthrough.R` with `message(...)` and wrap multi-arguments in `paste()`.
