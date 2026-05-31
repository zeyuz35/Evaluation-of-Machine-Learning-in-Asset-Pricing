## 2024-05-31 - Untrack IDE artifacts and binaries
**Learning:** Git tracks IDE artifacts (`.Rproj.user`, `.Rhistory`) and compiled binaries (`*.dll`) even if they are in `.gitignore` if they were added before the ignore rules were set.
**Action:** Use `git rm -r` to remove these tracked artifacts and ensure `.gitignore` rules prevent them from being tracked again.
