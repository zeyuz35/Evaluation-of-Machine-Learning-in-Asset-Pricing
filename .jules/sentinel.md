## 2024-05-13 - Remove Hardcoded AWS RSA Private Keys
**Vulnerability:** Found hardcoded RSA private keys (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) committed directly to the repository. This is a critical security vulnerability that exposes AWS resources to unauthorized access.
**Learning:** These files were likely committed by accident during the setup of an AWS EC2 instance. Secrets should never be committed to source control.
**Prevention:** The sensitive files have been physically deleted using `git rm` to remove them from version control and the filesystem. Ensure `*.pem` and `*.ppk` are added to a `.gitignore` to prevent accidental commits in the future, and use a secrets management system or environment variables instead of hardcoding credentials.
