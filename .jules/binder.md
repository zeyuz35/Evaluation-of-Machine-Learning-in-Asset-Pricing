## 2024-04-06 - Initial setup

**Learning:** Memory indicated: "When vendoring or bundling third-party R packages locally (e.g., a fork tracked in the repo), ensure auto-generated installation artifacts such as `Meta/`, `html/`, `help/`, compiled binaries (`libs/`), and lazy-load databases (`.rdb`, `.rdx`) are omitted from version control and explicitly ignored in `.gitignore`." and "Enforce repository artifact hygiene by ensuring `.gitignore` explicitly excludes IDE files (`*.Rproj`, `.Rproj.user/`, `.Rhistory`), build artifacts (`*.tar.gz`, `check/`, `..Rcheck/`, `00check.log`), and compiled binaries (`*.o`, `*.so`, `*.dll`, executables). Never commit these temporary artifacts to version control."
I found `R/xgboost` contains `libs/`, `html/`, `help/`, `Meta/`, and `.rdb`/`.rdx` files which are tracked. Also `.Rhistory`, `.Rproj.user`, and `.Rproj` exist and are tracked.

**Action:** Add exclusions to `.gitignore` and remove these generated artifacts from tracking.
