## 2024-05-18 - Replacing O(N^2) rbind loops in R with do.call(rbind, lapply(...))

**Learning:** In R, growing a dataframe row-by-row inside a `for` loop using `rbind()` causes an $O(N^2)$ performance penalty due to repeated memory reallocation and copy-on-modify semantics. This bottleneck is highly prevalent in manual grid-search tuning functions.

**Action:** Replace `for` loops that iteratively `rbind` dataframes with a vectorized `lapply` over the iteration indices, followed by a single `do.call(rbind, ...)` operation to concatenate the accumulated list of dataframes. This transforms the operation to $O(N)$ and is a standard performance improvement pattern in R.
