## 2024-05-30 - Remove hardcoded SSH private keys
**Vulnerability:** Found `rstudio.pem` and `rstudio.ppk` private keys committed to the repository in `R/AWS/`.
**Learning:** Committing private keys exposes infrastructure to unauthorized access and compromises security.
**Prevention:** Added `*.pem` and `*.ppk` to `.gitignore` to prevent accidental commits of private key files in the future.
