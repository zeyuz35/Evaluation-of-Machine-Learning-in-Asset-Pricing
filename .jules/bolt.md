## 2025-01-28 - Optimize loop subsetting in empirical robustness scripts
**Learning:** Using `dplyr::filter` inside `foreach` loops on large datasets adds significant overhead due to S3 dispatch, evaluation logic, and dataframe creation for each iteration. Base R logical subsetting `dataset[dataset$time == time_periods[t], ]` is considerably faster for single-condition row subsetting within iterative constructs.
**Action:** Replace `%>% filter(...)` with base R logical subsetting `[...]` in tight loop iterations across analysis scripts.
