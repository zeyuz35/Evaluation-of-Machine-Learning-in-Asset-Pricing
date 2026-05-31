## 2026-05-31 - Vectorize array subsetting in LSTM
**Learning:** Native array and matrix vectorization in R is significantly faster than using manual nested loops to iterate over dimension boundaries.
**Action:** Replace explicit loop assignments (like for loops iterating over dim) with direct vectorized slice assignments.
