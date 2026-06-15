## 2024-05-24 - Initial Setup
**Learning:** Initializing docent journal.
**Action:** Proceed with finding an issue.
## 2024-05-24 - Structured logging for progress bars
**Learning:** Replaced raw `cat()` used for progress bars with structured `message()`, specifically using `appendLF = FALSE` for the dots to prevent automatic newlines.
**Action:** Always convert all raw `cat()` calls for console progress tracking to `message()` consistently.
