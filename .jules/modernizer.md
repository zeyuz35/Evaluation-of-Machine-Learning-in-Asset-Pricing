## 2026-04-25 - Converting magrittr pipes to native pipes in dplyr chains

**Learning:** When modernizing `dplyr` pipelines (`filter`, `mutate`, `select`, etc.) in standalone R scripts, it is safe to replace magrittr pipes (`%>%`) with native pipes (`|>`) as long as the functions naturally accept the piped data as the first argument, and there is no use of the `.` placeholder.

**Action:** Before performing bulk `%>%` to `|>` conversions, check if the file uses `.` inside pipeline expressions. If not, standard text replacement (like `sed`) can safely modernize the file while reducing dependency overhead.
