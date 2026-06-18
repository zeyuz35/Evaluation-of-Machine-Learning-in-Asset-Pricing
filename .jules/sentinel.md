## 2024-06-18 - RStudio User Configuration Data Tracked in Repo
**Vulnerability:** Personal developer configuration and history files (e.g., `R/.Rhistory` and the `R/.Rproj.user/` directory) are committed and tracked in the repository, despite `.Rhistory` being in `.gitignore`.
**Learning:** `.gitignore` does not automatically untrack files that have already been committed to the index. A previous commit added these files, leading to continuous tracking of local machine paths, IDE configurations, and potentially sensitive command history.
**Prevention:** Use `git rm --cached` to correctly stop tracking user-specific configuration directories and history files that are correctly listed in `.gitignore` but incorrectly tracked by git.
