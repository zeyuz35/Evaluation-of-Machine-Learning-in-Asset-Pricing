## 2024-05-24 - Replace iterative rbind() with lapply() + do.call(rbind, ...)
**Learning:** In R, using `rbind()` iteratively inside a `for` loop to accumulate a data frame or matrix is an O(N^2) operation because it copies the entire object at each step. This creates a severe performance bottleneck.
**Action:** Replace `for` loop `rbind()` accumulation patterns with `lapply()` to build a list of elements, followed by a single `do.call(rbind, list)` or `dplyr::bind_rows(list)` operation at the end to assemble the final object efficiently.
## 2024-05-24 - Replace iterative rbind() with lapply() + do.call(rbind, ...)
**Learning:** In R, using `rbind()` iteratively inside a `for` loop to accumulate a data frame or matrix is an O(N^2) operation because it copies the entire object at each step. This creates a severe performance bottleneck.
**Action:** Replace `for` loop `rbind()` accumulation patterns with `lapply()` to build a list of elements, followed by a single `do.call(rbind, list)` operation at the end to assemble the final object efficiently.
