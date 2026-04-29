## 2024-05-18 - Untracked and auto-generated build artifacts

**Learning:** There are several auto-generated build artifacts and user-specific IDE files that are tracked in version control, like `R/xgboost/Meta/`, `R/xgboost/html/`, `R/xgboost/libs/`, `R/xgboost/help/`, and various `.Rhistory`/`.Rproj.user` files. Committing these creates messiness and they should be ignored.

**Action:** Remove these files from version control and add them to `.gitignore` to maintain proper repository artifact hygiene.
