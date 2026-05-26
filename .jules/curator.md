## 2024-05-23 - Fix silent type coercion in dataframe list-columns
**Learning:** Assigning a `list()` to a single element of an atomic numeric vector via single bracket `[i]` causes silent type coercion and flattens structured data.
**Action:** Initialize dataframe list-columns safely using `I(vector("list", N))` and assign elements iteratively using double brackets `[[i]]` without wrapping the value in a list.
