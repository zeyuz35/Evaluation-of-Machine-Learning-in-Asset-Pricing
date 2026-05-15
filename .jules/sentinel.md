## 2024-05-15 - Remove hardcoded AWS SSH private keys
**Vulnerability:** Found hardcoded AWS SSH private keys (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) committed directly to the repository. This is a critical security vulnerability as it could allow unauthorized access to the AWS EC2 instances associated with these keys.
**Learning:** The keys were likely generated and used for local development/testing of AWS SageMaker and EC2 integrations and mistakenly committed alongside the code.
**Prevention:** Remove the hardcoded keys from the repository using `git rm` to ensure they are deleted from disk, and add `*.pem` and `*.ppk` to `.gitignore` to prevent future accidental commits of private key files.
