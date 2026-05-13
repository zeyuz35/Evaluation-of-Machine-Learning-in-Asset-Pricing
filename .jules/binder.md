## 2024-05-13 - Installed Binary Packages in Git History
**Learning:** Found a fully compiled and installed binary version of `xgboost` (complete with `Meta/`, `R/*.rdx`, `libs/x64/xgboost.dll` etc.) committed inside `R/xgboost`. Committing installed binary packages is a major violation of package hygiene and CRAN standards (polluting the source tree, causing cross-platform errors, bloat).
**Action:** Remove the `R/xgboost` vendored binary package directory from version control completely to clean up the repository.
