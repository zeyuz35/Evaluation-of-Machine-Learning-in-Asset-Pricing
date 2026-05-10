## 2026-05-10 - Safe Initialization of List-Columns in R

**Learning:** When creating `data.frame`s intended for JSON serialization via `jsonlite`, initializing a column as numeric (e.g., `target = c(1:N)`) and later trying to insert arrays/lists into its rows (`df$target[i] <- list(array_data)`) triggers silent type coercion or an outright execution crash (`more elements supplied than there are to replace`).
**Action:** When a column is meant to hold nested data (lists or arrays), always initialize it explicitly as a list vector: `df$column <- vector("list", N)`.
