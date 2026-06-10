## 2024-06-10 - Vectorize loop assignments across time blocks in simulation code
**Learning:** Found deeply nested loops (iterating `1:N` and `1:Time`) in R simulation files (e.g., `gen_g_factor_panel`) that do element-wise assignments. Since these are mostly operations on 3D arrays sliced over time (`[, , t]`), they can be directly vectorized across the first dimension (N stocks).
**Action:** Replace `for (i in 1:N) { for (t in 1:Time) { ... } }` with `for (t in 1:Time) { ... }` and assign to the entire column/slice at once `[, , t]`. This speeds up the function significantly.
