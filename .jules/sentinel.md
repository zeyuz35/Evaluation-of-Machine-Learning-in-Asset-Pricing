## 2024-05-21 - Hardcoded SSH Private Keys
**Vulnerability:** The repository contains hardcoded SSH private keys (rstudio.pem and rstudio.ppk) in the R/AWS/ directory.
**Learning:** These files were likely committed by accident when setting up an AWS EC2 or SageMaker environment for RStudio.
**Prevention:** Always add *.pem and *.ppk to .gitignore before initializing or working in a repository that interfaces with cloud services via SSH.
