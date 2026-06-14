## 2024-06-14 - Replace legacy iterators
**Learning:** Legacy 1:length() and 1:nrow() patterns can cause 1:0 out-of-bounds errors on empty structures.
**Action:** Replace with safe seq_along() and seq_len() alternatives.
