## 2024-06-03 - Remove Hardcoded SSH Private Key
**Vulnerability:** A hardcoded SSH private key (`R/AWS/rstudio.ppk`) was found committed to the repository.
**Learning:** Hardcoded credentials and keys can easily leak access to infrastructure. It existed likely to facilitate easy logins during development.
**Prevention:** Never commit private keys. Add `.ppk`, `.pem`, and other key file extensions to `.gitignore` to prevent accidental commits. Use environment variables or secure key management systems for secrets.
