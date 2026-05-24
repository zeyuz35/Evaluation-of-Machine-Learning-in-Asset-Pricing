## 2024-05-23 - Remove tracked RStudio user artifacts
**Learning:** The `.Rproj.user` directory and `.Rhistory` file contain user-specific workspace state and command history. Even if listed in `.gitignore`, they remain tracked if added beforehand, causing unnecessary repository bloat and potential merge conflicts.
**Action:** Always use `git rm -r` to untrack and remove user-specific IDE artifacts while preserving the core `.Rproj` project definition.
