## YYYY-MM-DD - Hardcoded SSH Keys
**Vulnerability:** Found hardcoded SSH private keys `R/AWS/rstudio.pem` and `R/AWS/rstudio.ppk` checked into the repository.
**Learning:** These were likely used for testing AWS integration and accidentally committed.
**Prevention:** Remove keys from the repository and git history to prevent unauthorized access. Add them to `.gitignore` to prevent future commits. Use environment variables or secure key management systems for authentication instead.
