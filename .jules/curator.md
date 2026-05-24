## YYYY-MM-DD - [Title]
**Learning:** [Insight]
**Action:** [How to apply next time]

## 2025-05-24 - [Avoid Silent Type Coercion in R Data Frames]
**Learning:** Initializing a target variable column using `target = c(1:N)` makes it an atomic integer vector. In R, assigning a `list()` to a single element of an atomic vector (e.g., `df$target[i] <- list(val)`) causes silent type coercion and flattens the list.
**Action:** When creating a list-column within a `data.frame()` call, wrap the initialization in `I()` (e.g., `target = I(vector("list", N))`). When assigning elements iteratively, use double brackets `df$target[[i]] <- value`.
