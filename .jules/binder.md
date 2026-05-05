## 2024-05-24 - Initial Review

**Learning:** This repo is not a structured R package at all. It is a collection of scripts, RMarkdown files, and analysis notebooks for an honors thesis.

**Action:** Standard package hygiene steps (e.g., `R CMD check`, NAMESPACE) won't work. The best approach is to move top-level scripts containing `library()` calls into a subdirectory like `analysis/` or `scripts/` to improve structure and begin simulating an R package directory layout.
