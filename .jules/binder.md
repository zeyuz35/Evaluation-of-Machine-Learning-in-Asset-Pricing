## 2024-04-18 - Removed xgboost build artifacts from repository

**Learning:** The vendored `R/xgboost` package contained auto-generated installation artifacts (`Meta/`, `html/`, `help/`, `libs/`, and `.rdb`/`.rdx` files in `R/`) that were committed to the repository. These artifacts cause package hygiene and portability issues.

**Action:** Removed these artifacts from git cache and explicitly added them to `.gitignore` to prevent future commits. Also removed user-specific `.Rproj.user` and `.Rhistory` from source control.
