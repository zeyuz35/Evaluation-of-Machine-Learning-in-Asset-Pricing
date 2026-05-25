## 2024-05-18 - Convert print/cat output to structured message()
**Learning:** Raw `cat()` calls are used for diagnostic outputs (like error rates) in xgboost demos.
**Action:** Replace these raw `cat()` calls with structured `message()` calls using `paste()` or `paste0()`, and avoid manual newline additions as `message()` appends newlines automatically. Do not remove `print()` calls for dataframes/matrices/models as `message()` doesn't format them properly.
