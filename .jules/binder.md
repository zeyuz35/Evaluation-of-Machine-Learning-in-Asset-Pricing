## 2024-05-15 - Remove vendored binary packages
**Learning:** Vendoring fully installed binary packages (like xgboost with .dll or Meta/ files) in the repository pollutes the source tree and causes cross-platform errors.
**Action:** Always completely remove the directory of installed binary packages from version control, rather than selectively deleting internal files.
