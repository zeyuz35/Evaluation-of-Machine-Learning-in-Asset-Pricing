## 2024-05-14 - Optimize iterative rbind loops
**Learning:** Using `rbind` inside a `for` loop creates an O(N^2) bottleneck due to repeated memory reallocation and copying.
**Action:** Replaced iterative `rbind` with `do.call(rbind, lapply(...))` to accumulate results efficiently in `get_ELN_best_tune` and `get_RF_best_tune` functions.
