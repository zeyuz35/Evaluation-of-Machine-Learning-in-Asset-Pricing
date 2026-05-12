## 2024-05-13 - Replace base print() with message() in R
**Learning:** R demo scripts often use print() or cat() to output informational steps. The standard in R programming is to use message() for diagnostics and informational logging, as print() should be reserved for outputting values, and cat() for console formatting.
**Action:** Replace `print("info")` and `print(paste("var=", err))` calls with `message("info")` and `message(paste0("var=", err))` in the demo file `R/xgboost/demo/basic_walkthrough.R` to align with R's professional diagnostic message standard, which is under 50 lines.
