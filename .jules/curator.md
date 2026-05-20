## 2024-05-20 - Fix silent type coercion in dataframe list assignment
**Learning:** Assigning a `list()` to a single element of an atomic numeric vector (e.g., `df$target[i] <- list(...)` where `target` is initialized as `1:N`) causes silent type coercion and data flattening in R.
**Action:** To safely create list-columns within a `data.frame()` call, wrap the initialization in `I()` (e.g., `target = I(vector("list", N))`). When assigning elements iteratively, use double brackets `df$target[[i]] <- value`.
