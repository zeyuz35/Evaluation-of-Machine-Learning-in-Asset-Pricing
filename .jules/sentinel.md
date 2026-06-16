## 2024-05-24 - Remove Hardcoded SSH Private Key
**Vulnerability:** A hardcoded SSH private key (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) is committed to the repository.
**Learning:** Development tools and keys were likely added to the repository for convenience without considering the security implications of exposing private keys, especially if the repository is or becomes public.
**Prevention:** Use `.gitignore` to exclude sensitive files like `*.pem`, `*.ppk`, and `.env` files. Secrets should be managed securely (e.g., via AWS Secrets Manager or local uncommitted files) and never hardcoded or tracked in version control.
