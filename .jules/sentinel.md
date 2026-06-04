## 2024-06-04 - Remove hardcoded S3 bucket name
**Vulnerability:** Hardcoded S3 bucket names in R scripts (e.g. sagemaker-us-west-2-438078873022).
**Learning:** Hardcoding S3 bucket names can expose infrastructure details and create security/operational risks if the bucket changes or belongs to a specific account.
**Prevention:** Always use dynamic variables or configuration files for infrastructure references, like `session$default_bucket()`.
