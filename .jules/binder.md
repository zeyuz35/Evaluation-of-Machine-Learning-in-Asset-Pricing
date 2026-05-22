## 2024-05-23 - Remove tracked RStudio IDE artifacts
**Learning:** RStudio IDE artifacts like .Rproj.user, .Rproj, and .Rhistory were mistakenly tracked in version control, which bloats the repository.
**Action:** Use git rm -r to remove these artifacts from version control and ensure they are broadly ignored in .gitignore.
