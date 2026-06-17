## 2024-05-24 - Identifying missing file organization
**Learning:** Found several standalone scripts in `R/` that are not formal R files (e.g. `umap_play.R`).
**Action:** Need to determine if they should be moved to `analysis/` or similar, or just check the namespace imports.
## 2024-05-24 - Identifying exploratory scripts in R/
**Learning:** Exploratory/scratchpad scripts like `umap_play.R` should be moved to workflow-specific directories (e.g., `analysis/`) per the `agents.md` Stage 1 guidelines.
**Action:** Move `umap_play.R` to `analysis/umap_play.R` to maintain package directory hygiene.
