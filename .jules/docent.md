## 2024-06-11 - Replace cat with message for progress indicators and logs
**Learning:** Raw `cat()` output for progress logging or status messages does not play well with standard structured logging infrastructure or R's condition handling. The project uses it across Rmd files and xgboost demo files.
**Action:** Replace `cat()` with `message()`. For progress bar dots, use `message('.', appendLF = FALSE)`. For lines ending in `\n`, remove the `\n` in `message()` because it adds one automatically.
