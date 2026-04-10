## 2024-04-10 - Ignore Build Artifacts

**Learning:** The legacy vendored version of `xgboost` contains build artifacts such as `Meta/`, `html/`, `help/`, and compiled binaries in `libs/` that should not be tracked in version control.
**Action:** Add these paths to `.gitignore` and remove them from version control to maintain package hygiene.
