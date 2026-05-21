## 2024-05-20 - Remove vendored xgboost binary package and RStudio artifacts
**Learning:** Found a fully installed binary version of xgboost (with .dlls) and RStudio artifacts (.Rhistory, .Rproj.user) checked into the repository under R/, which causes pollution and violates CRAN structure.
**Action:** Removed vendored xgboost binary and RStudio artifacts via `git rm -r` to minimize dependency bloat and improve structure.
