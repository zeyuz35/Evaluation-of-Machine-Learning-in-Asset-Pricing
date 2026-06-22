## 2024-06-23 - Move exploratory script out of R directory
**Learning:** Standalone exploratory scripts like `umap_play.R` should not reside in the package `R/` directory as they clutter the core package namespace.
**Action:** Move exploratory scripts to `analysis/` or similar workflow-specific directories to maintain package hygiene.
