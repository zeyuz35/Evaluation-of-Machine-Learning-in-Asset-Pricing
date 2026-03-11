#' Coerce to matrix while preserving time-series classes and attributes
#'
#' @param x An object to coerce to matrix
#' @return A matrix that retains the original attributes but drops class to be computationally safe
as_matrix_preserve <- function(x) {
  # Capture original attributes
  orig_attrs <- attributes(x)

  # Coerce to matrix
  res <- as.matrix(x)

  # Restore attributes except those structural to matrices and class
  non_structural <- setdiff(names(orig_attrs), c("dim", "dimnames", "class"))
  for (attr_name in non_structural) {
    attr(res, attr_name) <- orig_attrs[[attr_name]]
  }

  res
}
