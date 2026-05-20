## 2024-05-20 - Remove RStudio IDE artifacts from version control
**Learning:** RStudio IDE artifacts (like .Rproj.user, .Rhistory, and .Rproj) were tracked in git, which clutters the repository with user-specific IDE configuration and history.
**Action:** Removed these IDE-specific files using git rm -r --cached and deleted them from the filesystem. Added *.Rproj to .gitignore.
