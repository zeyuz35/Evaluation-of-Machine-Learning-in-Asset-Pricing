## 2024-05-13 - Removed RStudio IDE artifacts

**Learning:** When addressing package hygiene, local IDE artifacts (such as `.Rhistory`, `.Rproj.user` directories, and `.Rproj` files) should be completely removed from tracking, as they add unnecessary bloat and lead to conflict. Furthermore, any `.Rproj` files must be thoroughly checked against `git ls-files` to ensure they are actually tracked before using `git rm --cached`, to avoid causing fatal git errors.

**Action:** Ensure these files are removed from the filesystem, untracked if they are in the tree, and added to the `.gitignore`. Use `git ls-files` to verify tracked status before running `git rm`.
