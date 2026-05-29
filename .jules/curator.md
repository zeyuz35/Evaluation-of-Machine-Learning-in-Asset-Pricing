## 2024-05-29 - Safe List-Column Assignment in R
**Learning:** Assigning a list to an element of an atomic numeric vector in a data.frame (e.g., `df$target[i] <- list(value)` where `target` is initialized as `c(1:N)`) causes silent type coercion and incorrectly flattened list data. This breaks downstream data serialization like JSON.
**Action:** When creating data.frames with list-columns, use `I(vector("list", N))` for initialization. When assigning, assign directly with `[[i]]` (e.g., `df$target[[i]] <- value` without `list()`).
