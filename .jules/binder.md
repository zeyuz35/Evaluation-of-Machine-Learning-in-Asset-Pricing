## 2024-05-24 - Do not vendor installed binary packages
**Learning:** Fully installed binary R packages containing `Meta/`, `.dll`, and `html/` were vendored in `R/xgboost`. This pollutes the repository, violates CRAN rules, and bloats the repo.
**Action:** Remove installed binary package directories entirely from the repository; dependencies should be managed via standard R mechanisms (like DESCRIPTION).
