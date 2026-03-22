## 2024-03-24 - Artifact tracking

**Learning:** Found installed build artifacts (e.g., `R/xgboost/` containing `Meta/` and `.dll` binaries) checked into version control, which pollutes the repository and violates package hygiene.

**Action:** Remove these unintended artifact commits from the repository and ensure `.gitignore` explicitly excludes compiled binaries (`*.dll`, `*.so`, `*.o`) to prevent future regressions.
