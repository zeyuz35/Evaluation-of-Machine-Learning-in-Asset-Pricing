## 2024-05-24 - Replace legacy iterators
**Learning:** Found widespread use of `1:length(x)` and `1:nrow(x)` which can cause out-of-bounds errors if the vector/dataframe is empty.
**Action:** Replace `1:length(x)` with `seq_along(x)` and `1:nrow(x)` with `seq_len(nrow(x))` across the codebase to ensure robustness.
