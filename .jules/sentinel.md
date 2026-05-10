## 2024-05-18 - Hardcoded AWS SSH Keys
**Vulnerability:** AWS SSH keys (rstudio.pem, rstudio.ppk) committed to version control.
**Learning:** Keys were likely added during environment setup/deployment without being added to .gitignore.
**Prevention:** Remove keys from repo, permanently add *.pem and *.ppk to .gitignore.
