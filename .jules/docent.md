## 2024-05-24 - Style fixes in xgboost demo
**Learning:** Found multiple instances of `cat()` calls and informal language with an emoji `:-)` in `R/xgboost/demo/create_sparse_matrix.R`.
**Action:** Replaced `cat()` calls terminating in `\n` with `message()` calls as per the project's codebase conventions on "Console Output and Messaging", and removed informal language and emojis to improve professional presentation and clarity.
