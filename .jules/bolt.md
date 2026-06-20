## 2024-05-18 - Replacing sequential `rbind` loops in grid search processing
**Learning:** Combining data frames sequentially inside a `for` loop with `rbind()` creates an $O(N^2)$ performance bottleneck due to memory reallocation and copying on every iteration. We found this pattern in `get_ELN_best_tune` and `get_RF_best_tune` functions across the Rmd files.
**Action:** Replace `for` loops updating a growing data frame with `lapply` combined with `do.call(rbind, ...)`. This builds a list in $O(N)$ time and then binds all elements efficiently in one pass.
