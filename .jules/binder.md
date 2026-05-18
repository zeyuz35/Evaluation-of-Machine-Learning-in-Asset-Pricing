## 2024-05-18 - Remove vendored binary packages and IDE artifacts
**Learning:** Vendored binary packages (e.g. xgboost with .dll files) and RStudio IDE artifacts pollute the version control, cause cross-platform issues, and violate CRAN package source standards.
**Action:** Remove installed binary packages and IDE artifacts directly via git rm.
