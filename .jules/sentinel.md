## 2024-06-20 - Hardcoded RSA Private Key Removal
**Vulnerability:** Found hardcoded AWS RSA private keys (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) in the repository.
**Learning:** Keys were likely used for development/testing access but accidentally committed.
**Prevention:** Use environment variables, a secret manager, or secure key storage outside of the version-controlled codebase to handle sensitive access credentials. Additionally, `.gitignore` should be updated to exclude `*.pem` and `*.ppk` files to prevent future accidental commits.
