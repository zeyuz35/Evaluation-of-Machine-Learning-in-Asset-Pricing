## 2024-05-01 - Initial setup
**Learning:** Initializing docent log.
**Action:** Proceed with style and formatting checks.
## 2024-05-01 - Replace raw cat with message in xgboost demo
**Learning:** Raw `cat()` commands are often used in R scripts to print formatted information but lack the formal structure of `message()` or actual logging tools. The `message()` function handles line breaks implicitly, preventing the need for `\n` while ensuring formal R logging output.
**Action:** Found a `cat()` call in `R/xgboost/demo/generalized_linear_model.R` outputting raw prediction errors. Replaced it with a formatted `message()` call and updated `!=` syntax for style adherence.
