## 2024-05-23 - Replace magrittr pipes with native R pipes
**Learning:** Legacy codebase used magrittr pipes (`%>%`), which requires importing the `magrittr` or `dplyr` package. Native R pipes (`|>`) eliminate this dependency for standard operations and are slightly faster as they are implemented in C inside base R.
**Action:** Replaced all `%>%` with `|>` in R scripts to modernize syntax and adhere to agents.md code modernization standards.
