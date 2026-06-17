## 2024-06-17 - Iteration modernizations
**Learning:** Legacy iteration patterns like `1:length(x)` and `1:nrow(df)` are prevalent in this repository, especially within R/simulation and R/empirical files. These patterns are unsafe if `x` or `df` are empty, resulting in `1:0` loops.
**Action:** Replace `1:length(x)` with `seq_along(x)` and `1:nrow(df)` with `seq_len(nrow(df))` across the codebase to adhere to Modernizer's primary goals of modernization and iterators efficiency/safety.
