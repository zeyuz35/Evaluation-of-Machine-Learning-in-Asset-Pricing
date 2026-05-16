## 2024-05-16 - Removed hardcoded AWS SSH keys
**Vulnerability:** Hardcoded AWS SSH private keys (.pem and .ppk) were committed to the repository in R/AWS/.
**Learning:** Sensitive files must be properly gitignored and never tracked in version control.
**Prevention:** Added *.pem and *.ppk to .gitignore and physically deleted the files using git rm.
