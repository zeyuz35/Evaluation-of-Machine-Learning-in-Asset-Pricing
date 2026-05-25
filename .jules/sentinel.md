## 2024-05-25 - Remove hardcoded SSH keys
**Vulnerability:** Hardcoded SSH private keys (rstudio.pem and rstudio.ppk) were committed to the repository in the R/AWS directory.
**Learning:** The keys were likely generated for setting up an AWS SageMaker instance and committed by accident during exploration.
**Prevention:** Add *.pem and *.ppk to .gitignore and never commit private keys.
