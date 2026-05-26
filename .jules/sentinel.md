## YYYY-MM-DD - Hardcoded SSH Keys Discovered
**Vulnerability:** Private SSH keys (rstudio.pem and rstudio.ppk) are hardcoded in the repository (R/AWS/rstudio.pem, R/AWS/rstudio.ppk). These were potentially used for the AWS Sagemaker demo setup.
**Learning:** Hardcoding credentials/secrets/keys in a codebase happens when testing scripts are checked into version control without a proper ignore rule or secret management workflow.
**Prevention:** Always use environment variables or a secure key management system for sensitive credentials, and add `*.pem` and `*.ppk` to `.gitignore`.
