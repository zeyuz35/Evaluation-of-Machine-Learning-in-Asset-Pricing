## 2024-05-11 - Safe Scaling of Time Series Objects
**Learning:** Calling `scale()` on time series objects (`ts`, `xts`, `zoo`) silently coerces them to plain matrices/arrays and strips their time series attributes (`tsp`, `index`, `class`), resulting in lost metadata.

**Action:** Created `safe_scale` to capture attributes prior to scaling and restore them afterward, avoiding data corruption.
