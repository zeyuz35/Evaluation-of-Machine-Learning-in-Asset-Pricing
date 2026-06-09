## 2024-06-09 - Remove Unused Dependencies in Exploratory Script
**Learning:** Found several unused `library()` imports (e.g., `purrr`, `tidyr`, `tsfeatures`, `gganimate`, `sneezy`) in an exploratory script (`R/umap_play.R`).
**Action:** Remove these unused dependencies to improve codebase hygiene and reduce dependency bloat, especially as `agents.md` emphasizes minimizing external dependencies.
