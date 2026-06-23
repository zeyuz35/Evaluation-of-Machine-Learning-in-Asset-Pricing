## 2024-05-24 - Preserving scaled attributes after cbind()
**Learning:** In R, applying `scale()` to a data.frame returns a matrix with attributes like `scaled:center` and `scaled:scale`. However, using `cbind()` to combine this matrix back into a data.frame silently drops these attributes in the resulting data.frame. This loses important scaling information.
**Action:** When using `cbind()` with a scaled matrix to create a dataframe, extract the attributes first and explicitly assign them back to the resulting data.frame to preserve metadata correctly.
