## 2024-05-24 - Fix silent type coercion in R data.frame list-columns
**Learning:** Assigning a `list()` to a single element of an atomic vector (e.g., initialized as `c(1:N)`) within a data frame causes silent type coercion and flattened data.
**Action:** Always wrap list-column initializations in `I(vector("list", N))` within `data.frame()` calls, and use double brackets `[[i]]` for iterative list assignment.
