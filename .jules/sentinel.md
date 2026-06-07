## 2024-05-24 - Removed AWS Private Keys
**Vulnerability:** Found AWS RStudio private keys (rstudio.pem and rstudio.ppk) hardcoded and tracked in the repository under R/AWS/.
**Learning:** These keys were likely tracked accidentally during development or setup of the AWS Sagemaker integration, as they allow SSH access to RStudio instances.
**Prevention:** Remove keys using git rm, add *.pem and *.ppk to .gitignore to prevent future keys from being tracked.
