## Modernizer Journal

## 2024-05-13 - Replace iterative rbind() with do.call(rbind, lapply()) in R

**Learning:** Iteratively calling `rbind()` inside a `for` loop in R is an $O(N^2)$ operation that is extremely slow because it repeatedly copies the growing dataframe in memory.
**Action:** Replaced `for` loop `rbind()` accumulation with `lapply` to collect dataframes in a list and combined them once using `do.call(rbind, list)`.
