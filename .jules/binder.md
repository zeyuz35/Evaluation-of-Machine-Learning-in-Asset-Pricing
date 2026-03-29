## YYYY-MM-DD - Binder

**Learning:** When vendoring or bundling third-party R packages locally (e.g., a fork tracked in the repo), auto-generated installation artifacts such as `Meta/`, `html/`, `help/`, compiled binaries (`libs/`), and lazy-load databases (`.rdb`, `.rdx`) should be omitted from version control and explicitly ignored in `.gitignore`. The R/xgboost directory in this repository contains build artifacts that have been checked into version control.

**Action:** Add these directories to `.gitignore` and run `git rm -r --cached` on the generated metadata directories to remove them from version control.
