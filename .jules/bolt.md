## 2024-05-18 - Optimized bind_rt_predictor
**Learning:** Found O(N^2) 'growing objects' performance bottleneck caused by iterative `rbind()` inside a `for` loop in `R/simulation/Simulation.Rmd` inside `bind_rt_predictor`.
**Action:** Replaced the `for` loop with `lapply` list accumulation followed by `do.call(rbind, df_list)`. This prevents redundant object reallocation.
