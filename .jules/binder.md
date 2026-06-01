## $(date +%Y-%m-%d) - Remove RStudio artifacts
**Learning:** RStudio creates `.Rhistory` and `.Rproj.user/` directories for user-specific settings which should not be tracked in version control. While `.Rproj.user` is in the `.gitignore`, its contents (and `.Rhistory`) have been committed.
**Action:** Remove these files from git history/tracking. Never track user-specific IDE settings.
