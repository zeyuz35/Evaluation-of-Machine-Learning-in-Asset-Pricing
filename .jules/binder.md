## 2024-05-24 - Enforce Explicit Namespacing
**Learning:** R scripts often contain global library() calls and magrittr pipes which can pollute the namespace.
**Action:** Remove global imports and use explicit pkg::function() syntax, transforming pipes into standard function calls to preserve functionality while adhering to the package hygiene standards.
