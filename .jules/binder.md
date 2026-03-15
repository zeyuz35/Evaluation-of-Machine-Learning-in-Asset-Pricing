## 2024-05-28 - Explicit Namespacing & File Organization

**Learning:** Exploratory scripts containing `library()` calls and hardcoded paths located in `R/` violate R package standards and cause `R CMD check` failures. These must reside in an `analysis/` or similar directory.

**Action:** Moved `R/umap_play.R` to `analysis/` and removed unused libraries. Replaced `%>%` with `|>` and fully qualified `dplyr::select()`. Always audit `R/` for non-function definitions.
