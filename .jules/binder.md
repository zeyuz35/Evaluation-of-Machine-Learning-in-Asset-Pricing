## 2025-03-25 - [Artifact Hygiene Improvement]

**Learning:** Found various auto-generated R package installation artifacts (Meta/, html/, help/, libs/) from a locally bundled version of xgboost, as well as RStudio IDE files (.Rproj.user/, .Rhistory) committed to version control. This pollutes the repository and goes against standard R package hygiene practices.

**Action:** Added explicit exclusion rules to `.gitignore` for these directories and files, and removed them from Git tracking to ensure the repository remains clean and only source files are tracked.
