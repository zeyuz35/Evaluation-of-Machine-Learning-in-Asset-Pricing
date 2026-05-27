## 2024-05-27 - [Vendor package cleanup]
**Learning:** Found a fully installed binary version of xgboost (with .dlls for windows in i386 and x64) checked into version control under `R/xgboost`.
**Action:** Do not vendor installed packages, particularly compiled extensions in version control, as they pollute the repo, bloat the size, and break cross-platform compatibility. I will remove it to clean up the codebase.
