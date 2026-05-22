## 2024-05-22 - Remove hardcoded SSH keys
**Vulnerability:** Found hardcoded SSH private keys (`rstudio.pem`, `rstudio.ppk`) in the `R/AWS/` directory.
**Learning:** These files were likely committed by accident when setting up an AWS EC2 or SageMaker instance and using them to connect via SSH/PuTTY.
**Prevention:** SSH keys and credentials should never be committed to source control. Use `.gitignore` and secure credential management solutions.
