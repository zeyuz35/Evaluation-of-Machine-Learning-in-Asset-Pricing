## 2024-05-20 - Removed hardcoded SSH keys
**Vulnerability:** Hardcoded SSH private keys (.pem and .ppk) were found in the repository.
**Learning:** Keys committed to source control expose the corresponding infrastructure to unauthorized access.
**Prevention:** Keys should be physically removed from the repository, and file extensions like *.pem and *.ppk should be added to .gitignore to prevent accidental commits.
