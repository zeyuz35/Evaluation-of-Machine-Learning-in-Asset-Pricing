## 2024-05-24 - Hardcoded SSH Keys Removal
**Vulnerability:** Hardcoded SSH private and public keys (`R/AWS/rstudio.pem`, `R/AWS/rstudio.ppk`) exist in the repository tree.
**Learning:** Development credentials and SSH keys used for AWS instance configuration and connections were unintentionally committed along with the AWS configuration scripts.
**Prevention:** Always use `.gitignore` to prevent committing `.pem`, `.ppk`, and `.key` files. Secrets should be securely managed via AWS Secrets Manager, SSM Parameter Store, or external credential helpers.
