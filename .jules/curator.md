## 2024-05-18 - Fix List Column Flattening
**Learning:** Assigning a list object (e.g., `list(c(5, 6))`) to a single element of an integer vector within a dataframe (e.g., `df$target[1] <- list(...)`) silently coerces the input into the vector's type, effectively flattening the structure or completely dropping the list wrapping.
**Action:** When initializing a dataframe with list columns, use `I(vector("list", N))` instead of standard numerical initializations. Additionally, to assign to an element of a list column, use double-bracket indexing `df$target[[1]] <- c(5, 6)`.
