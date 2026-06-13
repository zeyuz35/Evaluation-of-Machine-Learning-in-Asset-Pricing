## 2024-06-13 - Remove exposed SSH private keys
**Vulnerability:** Hardcoded SSH private keys (rstudio.ppk, rstudio.pem) were committed to the repository in the R/AWS/ directory.
**Learning:** These keys were likely used for authenticating with AWS resources during local development but committing them exposes sensitive infrastructure access.
**Prevention:** SSH keys and other secrets must never be committed; they should be added to .gitignore and handled securely via local configuration or secrets management.
