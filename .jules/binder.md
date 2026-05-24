## 2024-05-24 - Remove tracked RStudio IDE artifacts
**Learning:** RStudio IDE artifacts (`.Rproj.user`, `.Rhistory`, `.Rproj`) should never be tracked in version control, as they are user-specific and pollute the repository.
**Action:** Use `git rm -r` to remove them from version control and add them to `.gitignore` to prevent future tracking.
