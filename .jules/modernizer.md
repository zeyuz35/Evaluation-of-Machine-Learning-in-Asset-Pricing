## 2024-05-24 - [Replace 1:length(...) with seq_along(...)]
**Learning:** Legacy iterator syntax `1:length(x)` can cause errors when `x` has length 0, evaluating to `1:0`. Using `seq_along(x)` is the modern, idiomatic R pattern that safely handles 0-length vectors and clearly signals intent.
**Action:** Replaced instances of `1:length(...)` with `seq_along(...)` across all R and Rmd scripts to modernize loop patterns, improve robustness, and maintain existing performance logic without altering intended behavior.
