## 2024-05-20 - R Package Hygiene
**Learning:** Exploratory analysis scripts like `umap_play.R` should not reside directly in the root of an R package (`R/` directory). The `R/` directory is reserved for core package functions according to `agents.md`.
**Action:** Move exploratory scripts to a dedicated directory like `analysis/` or `scripts/` to maintain proper repository structure.
