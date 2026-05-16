## 2024-05-13 - Replace iterative rbind() with lapply() + do.call(rbind)
**Learning:** Iteratively appending to a dataframe inside a loop using `rbind()` leads to an O(N^2) memory reallocation bottleneck in R. Using `lapply()` to build a list and then calling `do.call(rbind, ...)` once is significantly faster and uses less memory.
**Action:** Replaced iterative `rbind()` implementations inside `get_ELN_best_tune` and `get_RF_best_tune` across multiple .Rmd files with the performant `lapply` + `do.call` pattern.
