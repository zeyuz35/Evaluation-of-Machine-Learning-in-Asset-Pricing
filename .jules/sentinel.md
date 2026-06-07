## 2024-06-07 - Remove private SSH keys
**Vulnerability:** Found hardcoded private SSH keys rstudio.pem and rstudio.ppk in the R/AWS/ directory.
**Learning:** These were likely accidentally committed when someone used them to connect to an RStudio EC2 instance.
**Prevention:** Do not add private SSH keys (*.pem, *.ppk) into version control. Ensure they are excluded in .gitignore.
