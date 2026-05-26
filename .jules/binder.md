## $(date +%Y-%m-%d) - Remove vendored xgboost binary
**Learning:** Found an entire compiled xgboost library (including dlls for x64/i386, HTML, Meta, libs, help files) within the `R/` directory. This is not the proper way to manage dependencies in an R repository, as binary installations should be handled via DESCRIPTION or standard R package management to prevent cross-platform issues and bloat.
**Action:** Removed the `R/xgboost` directory using `git rm -r` to clean up the repository structure and reduce bloat.
