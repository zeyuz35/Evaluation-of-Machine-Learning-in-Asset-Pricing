## 2026-03-27 - Artifact Cleanup

**Learning:** The vendored 'xgboost' package in 'R/xgboost' contained built R package artifacts (like 'Meta/', 'html/', 'help/', 'libs/', and compiled internal R environments in 'R/'). These should not be version controlled, as they are generated dynamically and depend on the execution environment.

**Action:** Removed the artifacts from tracking via 'git rm -r --cached' and added them to '.gitignore' to prevent them from being committed again.
