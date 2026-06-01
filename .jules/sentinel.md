## 2024-05-23 - Hardcoded RStudio Password in AWS CloudFormation Templates
**Vulnerability:** Found hardcoded password 'rstudio' used to create an rstudio user in `R/AWS/sagemaker_stack.txt` and `R/AWS/template1` EC2 UserData blocks.
**Learning:** Hardcoding passwords in infrastructure-as-code templates makes them visible to anyone who has access to the template and risks deploying instances with weak, known default passwords.
**Prevention:** Always use secure parameterization with `NoEcho: true` for passwords and secrets in CloudFormation templates to inject them securely at stack creation time.
