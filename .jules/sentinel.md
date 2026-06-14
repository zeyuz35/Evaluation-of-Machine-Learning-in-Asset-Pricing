## 2024-05-24 - Remove hardcoded SSH keys
**Vulnerability:** Found hardcoded SSH private keys R/AWS/rstudio.pem and R/AWS/rstudio.ppk committed to the repository.
**Learning:** SSH keys were likely generated for AWS EC2 instances and accidentally committed instead of being ignored or securely stored.
**Prevention:** Ensure that .pem and .ppk files are added to .gitignore and private keys are never committed.
