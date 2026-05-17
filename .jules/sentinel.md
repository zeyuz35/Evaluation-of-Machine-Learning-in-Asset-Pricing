## 2024-05-17 - Hardcoded SSH keys removal
**Vulnerability:** Hardcoded SSH keys (`rstudio.pem`, `rstudio.ppk`) were committed to the repository in the `R/AWS/` directory.
**Learning:** Hardcoded credentials and private keys should never be tracked in version control, as they expose systems to unauthorized access.
**Prevention:** Remove the files from git tracking, delete them from the file system, and update `.gitignore` to ignore `*.pem` and `*.ppk` files.
