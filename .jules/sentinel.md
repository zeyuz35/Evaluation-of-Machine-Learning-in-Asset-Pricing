## 2024-05-15 - [Remove Hardcoded SSH Keys]
**Vulnerability:** Found hardcoded SSH keys (rstudio.pem, rstudio.ppk) in the R/AWS/ directory committed directly to the repository.
**Learning:** Keys are often used for EC2 instance access (like Sagemaker in this context), and placing them in the same directory as scripts leads to accidental version tracking.
**Prevention:** Always add *.pem and *.ppk extensions to the .gitignore file prior to copying them to the repository, or keep them securely in ~/.ssh/.
