## $(date +%Y-%m-%d) - Replaced iterative `rbind` with `do.call(rbind, lapply(...))`
**Learning:** In R, growing a dataframe iteratively in a loop using `rbind` is a known performance bottleneck due to memory reallocation in each iteration.
**Action:** Replaced `for` loop `rbind` combinations with `do.call(rbind, lapply(...))` across codebase (`get_ELN_best_tune`, `get_RF_best_tune`).
