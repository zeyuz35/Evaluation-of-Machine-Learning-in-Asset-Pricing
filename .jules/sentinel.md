## 2024-06-02 - Removed hardcoded SSH private keys and binaries
**Vulnerability:** Found hardcoded SSH private keys (rstudio.pem, rstudio.ppk) and a windows executable binary (putty.exe) stored in version control under R/AWS/.
**Learning:** The files were likely added for convenience to simplify AWS SageMaker authentication, but version controlling secrets allows broad and permanent exposure. Binaries shouldn't be versioned.
**Prevention:** Always add secret keys, .pem, and .ppk files to .gitignore. Use environment variables, AWS KMS, or a secret manager to handle credentials.
