## 2026-05-04 - Vendored Installed Packages in Version Control

**Learning:** Vendoring an entire *installed* R package (like `xgboost` with its `libs/`, `Meta/`, and `.dll` binaries) rather than the package source pollutes the repository and is fundamentally broken across platforms, causing failures on Linux environments trying to load Windows `.dll` binaries.

**Action:** Remove the vendored, pre-compiled `xgboost` package entirely and update `.gitignore` to prevent its re-inclusion. This enforces CRAN standards that packages should list dependencies in DESCRIPTION rather than vendoring installed binary artifacts in the source tree.
