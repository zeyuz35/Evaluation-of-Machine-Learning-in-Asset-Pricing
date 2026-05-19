## 2024-05-19 - Remove Vendored Compiled xgboost Binary
**Learning:** Fully installed binary packages (containing `libs/`, `.dll`, `Meta/`, `.rdb`, etc.) vendored inside the repository pollute the source tree, cause cross-platform errors, and violate CRAN package structure guidelines.
**Action:** Always completely remove directories containing installed binary packages from version control, relying instead on standard R dependency management (e.g., `DESCRIPTION`) to provide the required packages.
