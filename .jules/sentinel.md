## 2024-05-24 - Remove RSA Private Keys and SSH Client Config Files
**Vulnerability:** Found `rstudio.pem` (RSA private key), `rstudio.ppk` (PuTTY Private Key), and `putty.exe` committed to the repository in the `R/AWS/` directory.
**Learning:** These files are sensitive authentication tokens to access AWS/RStudio resources. They should never be tracked in a source control system, as it compromises the system's security. They were likely accidentally committed alongside other AWS setup scripts.
**Prevention:** Do not check in `.pem`, `.ppk`, or `.exe` files. Add these file extensions to `.gitignore` to prevent future accidental commits.
