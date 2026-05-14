## Modernizer Journal
## 2024-05-14 - Replace rbind in loop with do.call(rbind, lapply)
**Learning:** Replaced `rbind` in a `for` loop with `lapply` and `do.call(rbind, ...)` for `get_ELN_best_tune` and `get_RF_best_tune` across multiple Rmd files. However, parsing the files using `knitr::purl` can fail due to duplicate chunk labels like 'elastic_net' in the Rmd files. We can't rely on `knitr::purl` passing if there are duplicate chunk labels pre-existing.
**Action:** Use `git diff` to manually verify changes instead of failing if `knitr` fails to parse duplicate chunks.
