## 2024-05-11 - Hardcoded SSH Keys Removed

**Vulnerability:** Found hardcoded SSH private keys `R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk` committed to the repository.
**Learning:** These files are used for accessing AWS environments but were inadvertently included in version control.
**Prevention:** Use `.gitignore` to prevent `.pem` and `.ppk` files from being tracked by Git. Private keys should be securely injected via environment variables or managed through a secret manager.
