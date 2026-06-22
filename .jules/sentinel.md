## 2024-05-24 - Hardcoded SSH Private Key Removal
**Vulnerability:** Found a hardcoded RSA private key (`R/AWS/rstudio.pem`) in the repository.
**Learning:** Development or test private keys are sometimes committed by mistake, allowing unauthorized access to infrastructure if the key is still valid or reused elsewhere.
**Prevention:** Always add `.pem`, `.key`, and similar sensitive file extensions to `.gitignore` and use environment variables, AWS Systems Manager Parameter Store, or AWS Secrets Manager for secret injection instead of hardcoding credentials.
