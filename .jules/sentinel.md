## 2024-05-18 - Remove hardcoded AWS private keys
**Vulnerability:** Found hardcoded AWS SSH private key files (rstudio.pem and rstudio.ppk) checked into version control under R/AWS/.
**Learning:** Checking in private keys exposes credentials to anyone who has access to the repository, potentially compromising cloud infrastructure.
**Prevention:** Private keys must never be committed. Added *.pem and *.ppk to .gitignore and physically removed the files via git rm.
