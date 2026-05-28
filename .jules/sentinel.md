## 2024-05-15 - Remove hardcoded SSH keys
**Vulnerability:** Found `R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk` checked into version control. These are RSA private keys.
**Learning:** These files were likely committed by accident during the development of an AWS integration or EC2 deployment script (`rstudio`). They pose a severe security risk if the repository is public or accessed by unauthorized users.
**Prevention:** Add `*.pem` and `*.ppk` to `.gitignore` to prevent future accidental commits of private keys. Use external secrets management or AWS KMS instead of committing keys.
