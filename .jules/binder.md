## 2024-05-28 - Remove tracked RStudio user artifacts
**Learning:** RStudio user artifacts like .Rhistory and .Rproj.user directories should never be tracked in version control, as they are user-specific and pollute the repository.
**Action:** Always check for and remove tracked .Rhistory files and .Rproj.user directories when auditing R projects.
