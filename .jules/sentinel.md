## 2024-05-24 - Exposed SSH Keys in Source Control
**Vulnerability:** Private SSH keys (`rstudio.pem` and `rstudio.ppk`) were committed to the repository in the `R/AWS/` directory.
**Learning:** SSH keys and other credentials are often inadvertently included when developers copy their working environment or configuration directories into the project tree.
**Prevention:** Never commit private keys. Always include `*.pem`, `*.ppk`, and similar credential extensions in `.gitignore`.
