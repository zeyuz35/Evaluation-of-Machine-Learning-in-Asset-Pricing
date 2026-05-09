## 2024-05-09 - Hardcoded AWS SSH Keys
**Vulnerability:** Found hardcoded `rstudio.pem` and `rstudio.ppk` private keys in the `R/AWS/` directory tracked by git.
**Learning:** These files are likely remnants of the author testing AWS connections or deploying an instance.
**Prevention:** These keys must be untracked using `git rm --cached` and ignored in `.gitignore` to prevent leaking SSH access to cloud instances.
