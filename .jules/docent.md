## 2024-05-17 - [Initial Docent]
**Learning:** Initial setup
**Action:** Initial setup
## 2024-05-17 - [Convert raw cat to structured message in Neural Network progress callbacks]
**Learning:** Found several callbacks for keras that use `cat("\n")` and `cat(".")` to print progress. In R, it is best practice to use `message()` for structured logging instead of raw `cat()`.
**Action:** Replaced `cat("\n")` with `message("")` and `cat(".")` with `message(".", appendLF = FALSE)` in `R/empirical/Real Data.Rmd`, `R/simulation/Simulation Models.Rmd` and other robustness check `.Rmd` files. This aligns with standard R logging practices.
