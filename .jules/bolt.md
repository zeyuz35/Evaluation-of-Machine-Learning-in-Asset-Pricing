## 2024-05-24 - R rbind in loop optimization

**Learning:** Growing data structures (like dataframes) inside a loop using `rbind()` forces R to copy the entire object in memory during every iteration. This results in $O(N^2)$ time complexity and is a massive performance bottleneck when processing large datasets or iterating over large hyperparameter grids.
**Action:** Always refactor `df <- rbind(df, df_new)` inside loops. Instead, pre-allocate a list (`list_dfs <- vector("list", N)`), store the components in the list during the loop (`list_dfs[[i]] <- df_new`), and then combine them once outside the loop using `df <- do.call(rbind, list_dfs)`. This simple pattern provides $O(N)$ execution time and significant performance boosts.
