## 2025-02-26 - Hardcoded AWS passwords
**Vulnerability:** AWS EC2 templates (sagemaker_stack.txt and template1) contained hardcoded default passwords for the rstudio user.
**Learning:** Hardcoded passwords in infrastructure templates expose instances to unauthorized access if the templates or resulting instances are accessible.
**Prevention:** Always use secure, dynamically provided parameters with NoEcho: true for passwords in infrastructure-as-code templates.
