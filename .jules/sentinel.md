## 2024-05-18 - Remove hardcoded SSH keys
**Vulnerability:** Private SSH keys (`R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk`) were hardcoded and committed into the repository, exposing the keys and risking unauthorized access.
**Learning:** Keys were likely added for convenience during development/deployment on AWS without being excluded via `.gitignore`.
**Prevention:** Add explicit exclusions in `.gitignore` for standard private key extensions (`*.pem`, `*.ppk`, `id_rsa`, etc.) and ensure tools/processes rely on environment variables or securely managed secrets (e.g., AWS Secrets Manager or SSH agents) rather than repository files. Use a `.gitignore` scanner or secret-finding tool (like `git-secrets`) locally before committing.
