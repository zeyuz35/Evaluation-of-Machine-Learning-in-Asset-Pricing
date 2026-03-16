## 2024-05-18 - R CMD check in unstructured repository

**Learning:** When acting as Binder in a repository that does not have a standard R package structure (missing DESCRIPTION, tests/, etc. in root), running `R CMD check .` will fail with "File DESCRIPTION does not exist", creating a `..Rcheck` artifact folder. You cannot strictly enforce a 0-NOTE check in such unstructured environments, but you still need to ensure standard package hygiene rules (like moving exploratory scripts out of `R/`) are followed.

**Action:** Clean up `.Rcheck` or `..Rcheck` directories explicitly after a check fails or completes, and do not commit them. Recognize that `R CMD check` requires a formal package structure.
