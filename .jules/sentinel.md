## 2024-05-30 - Remove hardcoded AWS SSH keys
**Vulnerability:** Found hardcoded RSA private keys (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) tracked in the repository.
**Learning:** SSH keys used for accessing AWS instances were accidentally committed to version control, exposing infrastructure access.
**Prevention:** Add `*.pem` and `*.ppk` to `.gitignore` to prevent private key files from being tracked in the repository.
