## 2024-05-18 - Move exploratory script out of package directory
**Learning:** `agents.md` explicitly states that exploratory or scratchpad scripts must not reside in the package `R/` directory to maintain package hygiene. `R/umap_play.R` is an exploratory script that was residing in the `R/` directory.
**Action:** Move exploratory scripts to a workflow-specific directory such as `analysis/` using `git mv` or `mkdir -p analysis && git mv <file> analysis/`.
