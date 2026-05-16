## 2024-05-16 - Prevent Silent Data Flattening in List-Columns
**Learning:** In R, explicitly initializing list-columns in a dataframe using `target = I(vector("list", N))` and assigning elements via `[[i]]` prevents the silent type coercion and data flattening that occurs when assigning lists to scalar vectors (e.g., `target = c(1:N)` and `df$target[i] <- list(...)`).
**Action:** Use `I(vector("list", length))` within `data.frame()` and `[[i]]` assignment to guarantee list structure preservation during iterative assignment, particularly before JSON serialization.
