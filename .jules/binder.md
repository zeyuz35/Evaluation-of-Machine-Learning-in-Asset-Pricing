## 2024-06-12 - Remove User Artifacts
**Learning:** RStudio creates user-specific artifacts (e.g., `.Rhistory` and `.Rproj.user/`) that should not be tracked in version control, as this violates artifact hygiene standards and clutters the repository.
**Action:** Always identify and remove `.Rproj.user/` and `.Rhistory` directories/files and ensure they are untracked via `.gitignore`.
