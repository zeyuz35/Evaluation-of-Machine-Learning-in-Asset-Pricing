## 2024-05-24 - Optimize get_ELN_best_tune function
**Learning:** In R, dynamically growing a dataframe/matrix within a for loop using `rbind` scales poorly (O(N^2)) due to repeated memory allocations and copying. This pattern was found in `get_ELN_best_tune` in multiple Rmd files.
**Action:** Replace `for (i in ...) rbind(...)` with `do.call(rbind, lapply(..., cbind(...)))` to allocate memory once and dramatically improve performance.
