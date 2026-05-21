## Modernizer Journal
## 2024-05-24 - Optimize Iterative rbind in R

**Learning:** An extremely common but inefficient pattern in legacy R code is to iteratively grow a data frame or list using `df <- rbind(df, new_row)` inside a `for` loop. This results in $O(N^2)$ memory reallocation, heavily bottlenecking performance on large loops. This repository relied on it heavily in tuning functions like `get_ELN_best_tune` and `get_RF_best_tune`.

**Action:** Replace `for` loops containing an iterative `rbind` with a list accumulation using `lapply` (or `mclapply`/`future_lapply` if parallel) and merge the final list of results outside the loop exactly once using `do.call(rbind, list_name)`. This provides significant performance benefits without sacrificing the structure or readability. When applying mass replacements using Python regex tools, target specific functions (e.g. `get_ELN_best_tune`) and strictly verify the file content differences to ensure safety.
