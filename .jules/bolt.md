## 2024-05-13 - Performance Improvements

**Learning:** Use of `dplyr::filter` in `for` loops on large dataframes creates performance bottlenecks compared to base R subsetting or functional methods.

**Action:** Replace `dplyr::filter` with base R vectorised subsetting in loop iterators to increase performance, specifically when iterating over distinct groups or slicing multiple times.
