## 2024-05-27 - Replace magrittr pipes and loops
**Learning:** Found old `%>%` usage in `R/AWS/sagemaker_run.R` and `for(i in 1:length(x))` in `R/AWS/sagemaker_run.R` and `R/AWS/sagemaker.R`. According to `agents.md`, need to replace `%>%` with `|>` and rename `i` to `ii` or use `future_apply` or similar functional frameworks where possible.
**Action:** Replace `%>%` with `|>` and update the single character iterators in `R/AWS/sagemaker_run.R` and `R/AWS/sagemaker.R` to double characters, and/or convert to `lapply`/`future_lapply`.
## 2024-05-27 - Iterator Modernization
**Learning:** Found loops using `1:length(x)` and `i` as the loop counter. According to agents.md, `i` should be renamed to double characters like `ii`, and it's best practice to use `seq_along(x)` instead of `1:length(x)` to avoid errors when `length(x)` is 0. Also, `1:10` loops should be refactored to `seq_len(10)` or `seq_along` to be safer.
**Action:** Replace single character iterators `i` with `ii`, and update loop ranges to use `seq_len()` or `seq_along()`. Always ensure related variables like `i` are updated in the loop body to `ii`.
## 2024-05-27 - Package Source Modification
**Learning:** Do not manually edit auto-generated package files like `NAMESPACE` and `DESCRIPTION` unless you are certain you have also refactored all underlying upstream `.R` files (especially those containing `roxygen2` comments) that generate them.
**Action:** When updating package dependencies or removing syntax imports, strictly perform changes in `R/` package source files, then let the package build toolchain re-generate the artifacts. If no upstream sources contain the target syntax, leave the artifacts alone unless you have explicit directives to reconstruct the whole package structure.
