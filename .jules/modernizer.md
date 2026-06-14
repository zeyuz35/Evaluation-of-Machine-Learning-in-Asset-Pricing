## 2024-06-15 - Modernizer init
**Learning:** Found multiple instances of `1:length(x)` and `1:nrow(x)` which are legacy iterator anti-patterns in R, susceptible to the "1:0 bug" (where an empty object produces `c(1, 0)` and causes out-of-bounds indexing). Modern robust equivalents are `seq_along(x)` and `seq_len(nrow(x))`.
**Action:** Replace `1:length(x)` and `1:nrow(x)` with `seq_along(x)` and `seq_len(nrow(x))` across the codebase to ensure iterator safety without changing functionality.
