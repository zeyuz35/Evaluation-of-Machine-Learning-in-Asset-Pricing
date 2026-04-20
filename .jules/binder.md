## 2024-04-20 - [Fix R package artifacts in git]

**Learning:** [Vendored R packages and RStudio create multiple local cache files, build artifacts (Meta/, help/, html/, libs/), and IDE-specific directories (.Rproj.user, .Rhistory) that should not be tracked in version control, as they are auto-generated or compiled machine-specific files.]

**Action:** [Explicitly un-track and ignore generated R package metadata, IDE files, and compiled libraries from Git tracking when working with a locally bundled legacy R package structure.]
