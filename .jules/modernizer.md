## 2024-05-15 - Rbind performance in bind_rt_predictor
**Learning:** `rbind()` inside a loop exhibits O(N^2) complexity in R because memory is reallocated each time.
**Action:** Replaced `rbind` loop with `lapply` list accumulation followed by `do.call(rbind, ...)`.
