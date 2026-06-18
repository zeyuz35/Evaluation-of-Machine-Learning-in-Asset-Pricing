## 2026-06-18 - Move standalone exploratory script out of R/
**Learning:** The memory states "According to the project's agents.md Stage 1 guidelines, exploratory or scratchpad scripts (e.g., standalone analysis files) must not reside in the package R/ directory. They should be moved to workflow-specific directories such as analysis/ to maintain package directory hygiene." The file R/umap_play.R is clearly an exploratory script.
**Action:** Created analysis/ directory and used `git mv R/umap_play.R analysis/umap_play.R` to satisfy the repository constraints without breaking anything.
