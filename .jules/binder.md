## 2024-04-07 - Artifact Hygiene Check

**Learning:** The legacy `xgboost` vendor package and `R/.Rproj.user` files are mistakenly tracked in the git repository. We need to enforce repository artifact hygiene by ensuring `.gitignore` explicitly excludes IDE files (`.Rproj.user/`, `.Rhistory`) and compiled binaries (`*.dll`).

**Action:** Update `.gitignore` and remove these unwanted artifacts using `git rm --cached`.
