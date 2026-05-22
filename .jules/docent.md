## 2024-11-20 - Replace raw cat() calls with structured message() in demos
**Learning:** Using raw `cat()` calls for logging output creates unstructured text and is not standard practice for informational messages in R scripts.
**Action:** Always replace informational `cat()` calls with `message()`, ensuring multiple arguments are concatenated with `paste0()` and trailing newlines are removed, as `message()` appends newlines automatically.
