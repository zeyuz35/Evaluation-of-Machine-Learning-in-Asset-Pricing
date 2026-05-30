## 2024-05-30 - Remove untracked RStudio artifacts from version control
**Learning:** RStudio user-specific artifacts (such as .Rhistory files and .Rproj.user/ directories) can cause merge conflicts, bloat the repository size, and should not be tracked in version control, as established by memory and standard best practices.
**Action:** Removed these files using git rm --cached to clean up the repository. Ensure these paths are also properly managed by .gitignore for new R projects.
