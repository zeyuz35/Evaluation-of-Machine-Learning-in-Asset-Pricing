## 2024-05-24 - Artifact Hygiene
**Learning:** Repositories often accidentally track user-specific artifacts (like .Rhistory or .Rproj.user/) and compiled binaries (.dll, .exe), which bloat the repository and can cause platform-specific issues or security risks.
**Action:** Always verify tracked files against a strict .gitignore and ensure binaries and personal environment files are removed from the git index and excluded.
