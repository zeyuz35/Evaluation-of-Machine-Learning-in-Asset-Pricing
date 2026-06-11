## 2024-06-11 - Replace legacy iterators with modern safe alternatives
**Learning:** Legacy iterators like `1:length(x)` and `1:nrow(x)` can cause 1:0 bounds errors when collections are empty.
**Action:** Replace `1:length(x)` with `seq_along(x)` and `1:nrow(x)` with `seq_len(nrow(x))` for safer code.
