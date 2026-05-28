## 2024-05-28 - Exposed SSH Private Keys
**Vulnerability:** Found hardcoded SSH private keys (`rstudio.pem` and `rstudio.ppk`) and an executable (`putty.exe`) in the `R/AWS/` directory.
**Learning:** These files were likely committed by mistake during the setup of an AWS EC2 instance for SageMaker experiments, exposing the instance to unauthorized access.
**Prevention:** Always use `.gitignore` to exclude sensitive files such as `*.pem`, `*.ppk`, and credentials. Manage secrets via environment variables or secure secret managers. Never commit private keys to version control.
