## 2024-05-18 - Replacing `cat()` with `message()` in progress bars
**Learning:** `cat()` is often used to display progress bars (dots and newlines) when training neural networks in R using keras. The project's style standards dictate replacing `cat()` used for progress with `message()`.
**Action:** Replace `cat("\n")` with `message("")` and `cat(".")` with `message(".", appendLF = FALSE)` inside the R code.
