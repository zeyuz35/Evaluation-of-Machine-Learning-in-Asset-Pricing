## 2024-05-18 - Hardcoded SSH Keys
**Vulnerability:** Hardcoded AWS SSH private keys (.pem and .ppk files) were committed to the repository in the R/AWS/ directory.
**Learning:** Hardcoded secrets left in the repository present a critical security risk as they allow unauthorized access to the associated AWS instances.
**Prevention:** Remove sensitive key files from version control and the file system, and update .gitignore to prevent their accidental inclusion in the future.
