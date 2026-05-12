## 2024-05-12 - Resolve O(N^2) rbind copying
**Learning:** Found multiple instances where `rbind` was used repeatedly inside a `for` loop, causing an O(N^2) performance bottleneck due to copying at every iteration.
**Action:** Replaced repetitive `rbind` inside loops with list accumulation (e.g., `tune_list[[i]] <- ...`) followed by a single `do.call(rbind, tune_list)` call to significantly improve script execution time.
