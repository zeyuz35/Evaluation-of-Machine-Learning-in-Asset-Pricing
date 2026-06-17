## 2024-06-17 - Modernize Iterators
**Learning:** Legacy `1:length(x)` and `1:nrow(df)` iterators are pervasive and susceptible to out-of-bounds errors when objects are empty.
**Action:** Replace `1:length(x)` with `seq_along(x)` and `1:nrow(df)` with `seq_len(nrow(df))` across the codebase to ensure robust iteration.
