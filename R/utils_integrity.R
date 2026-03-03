# Section -----
#' Preserve Attributes during Matrix Coercion
#'
#' \code{as_matrix_preserve} safely coerces objects to matrix.
#' It ensures scaling and transformation attributes are retained.
#'
#' @param x An object to be coerced to a matrix.
#' @param ... Additional arguments passed to \code{as.matrix}.
#'
#' @return A matrix with non-structural attributes preserved.
#' @export
as_matrix_preserve <- function(x, ...) {
  stopifnot(!is.null(x))

  original_attrs <- attributes(x)
  mat <- as.matrix(x, ...)

  if (!is.null(original_attrs)) {
    if (isS4(x)) {
      return(mat)
    }

    excluded <- c("dim", "dimnames", "names", "class")
    safe_attrs <- original_attrs[setdiff(names(original_attrs), excluded)]

    if (length(safe_attrs) > 0) {
      attributes(mat) <- utils::modifyList(attributes(mat), safe_attrs)
    }
  }

  mat
}
