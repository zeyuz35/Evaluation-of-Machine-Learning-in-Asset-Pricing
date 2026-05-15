## 2024-05-24 - Replace iterative rbind in loops with lapply
**Learning:** Iterative rbind in loops causes O(N^2) memory reallocation performance issues in R.
**Action:** Use lapply to construct a list of objects and then use do.call(rbind, ...) to bind them once at the end.
