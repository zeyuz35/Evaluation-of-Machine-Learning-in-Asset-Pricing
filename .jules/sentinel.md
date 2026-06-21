## 2024-06-21 - Remove Hardcoded RStudio Passwords
**Vulnerability:** Hardcoded `rstudio` passwords and `passwd --stdin` usage found in CloudFormation templates (`R/AWS/sagemaker_stack.txt`, `R/AWS/template1`).
**Learning:** Hardcoding credentials in version control allows unauthorized access to spun-up RStudio servers.
**Prevention:** Always use CloudFormation parameters with `NoEcho: true` for sensitive inputs, and prefer `chpasswd` over `passwd --stdin` for better compatibility.
