## 2024-05-15 - R message() formatting
**Learning:** When replacing raw `cat(".")` output with `message()`, `message(".", appendLF = FALSE)` accurately replicates the inline behavior of `cat(".")` for progress indications. Also, empty `cat("\n")` should be replaced with `message("")`. `knitr::purl()` might fail if `.Rmd` chunks have duplicated chunk names.
**Action:** Use these exact replacements in R packages/scripts. Use manual static syntax validation if `knitr::purl` encounters syntax or metadata issues.
