## 2025-05-31 - Remove Hardcoded RSA Private Keys
**Vulnerability:** Found unencrypted RSA private keys (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) hardcoded directly in the repository.
**Learning:** These files appear to be SSH keys for connecting to AWS EC2 instances provisioned via a CloudFormation template. It is dangerous to commit private keys to version control as anyone with access to the repo can use them.
**Prevention:** Always add `.pem` and `.ppk` files to `.gitignore` and manage SSH keys securely (e.g., via AWS Secrets Manager or injecting at runtime) rather than checking them into the repository.
