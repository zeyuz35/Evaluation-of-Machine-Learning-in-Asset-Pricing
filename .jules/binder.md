## 2024-05-17 - [Remove Vendored Binary Packages]
**Learning:** Fully installed binary packages (like xgboost with .dll or Meta/ directories) should never be vendored in a repository. They pollute the source tree, cause cross-platform errors, and violate CRAN standards.
**Action:** Always identify and completely remove the directory of installed binary dependencies rather than individually modifying auto-generated files.
