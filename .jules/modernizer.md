## Modernizer Journal
## 2024-04-10 - Optimizing covariance calculations with tcrossprod

**Learning:** `(Lambda) %*% t(Lambda)` is a common pattern for calculating covariance matrices but is inefficient compared to base R's built-in `tcrossprod(Lambda)`. However, `tcrossprod` is less immediately mathematically clear to casual readers than the explicit matrix multiplication.

**Action:** Replace `(X) %*% t(X)` patterns with `tcrossprod(X)` for improved matrix multiplication performance, but strictly ensure that an explanatory comment (e.g., `# Equivalent to (X) %*% t(X) but faster`) is added to maintain the readability and explicitly document the intent behind the optimization.
