## 2024-06-07 - Replace 1:length and 1:nrow with seq_along and seq_len
**Learning:** R loops using `1:length(x)` or `1:nrow(x)` can fail or produce unexpected results if the length is 0 (resulting in a sequence like `1:0`). R's modern, idiomatic approach is to use `seq_along(x)` and `seq_len(nrow(x))` to safely iterate over vectors and rows. This improves both robustness (avoiding the 1:0 bug) and clarity.
**Action:** Replace `1:length(variable)` with `seq_along(variable)` and `1:nrow(variable)` with `seq_len(nrow(variable))` throughout the `R/` directory.
