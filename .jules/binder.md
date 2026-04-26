## 2026-04-26 - R Directory Hygiene

**Learning:** Exploratory scripts containing top-level `library()` calls and unstructured analysis code were mixed in the `R/` directory. The R package standard dictates that `R/` should exclusively contain function definitions. Additionally, vendored packages might carry auto-generated artifacts (`Meta/`, `html/`, etc.) that shouldn't be in version control.

**Action:** Moved the standalone exploratory script `umap_play.R` out of `R/` into an `analysis/` directory to satisfy package structure requirements, and added `analysis` to `.Rbuildignore`. Removed transient vendored artifacts from git tracking and ignored them in `.gitignore`.
