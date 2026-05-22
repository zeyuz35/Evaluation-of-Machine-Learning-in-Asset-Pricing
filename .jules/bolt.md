## 2024-05-22 - Replaced iterative rbind with do.call
**Learning:** In R, using `rbind` inside a `for` loop dynamically grows a data frame at each step, which causes O(N^2) memory reallocation and represents a significant performance bottleneck.
**Action:** Replace `for` loop `rbind` accumulations with `lapply` to create a list, followed by a single `do.call(rbind, ...)` call to drastically improve execution speed.
