## Modernizer Journal
## 2024-05-26 - [Modernizer: Iterator Syntax Update]
**Learning:** Legacy codebase used `1:length(x)` and `1:N` iterators which are unsafe (they can produce `1:0` loops). Modern R prefers `seq_along(x)` or `seq_len(N)`.
**Action:** Replaced instances of `1:length(x)` and `1:N` iterators with `seq_along()` and `seq_len()` in loops and indices using a targeted string replacement script.
