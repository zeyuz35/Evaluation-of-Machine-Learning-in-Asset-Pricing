## 2024-06-08 - Use message() instead of cat() for informative logs in R
**Learning:** Found instances where `cat("\n")` and `cat(".")` were used as a crude progress bar. The agents.md specifically states "Utilize packageStartupMessage() within .onAttach() for package load notifications, rather than cat() or standard message()". It also says under 1. 🔍 REVIEW - Inspect documentation and style: "Raw output (print/cat) that should be structured logging or messages."
**Action:** Replaced `cat("\n")` with `message("")` and `cat(".")` with `message(".", appendLF = FALSE)`
