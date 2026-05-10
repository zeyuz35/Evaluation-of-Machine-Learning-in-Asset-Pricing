## 2024-05-10 - Removed installed binary R package
**Learning:** Installed binary R packages (containing libs/, Meta/, etc.) should never be committed to version control as they pollute the repo with binaries and break cross-platform compatibility.
**Action:** Always completely remove vendored installed binary packages (like R/xgboost) from the repository using git rm -r to maintain package hygiene.
