#' Coerce to matrix while preserving attributes and class
#'
#' @param x An object to coerce to a matrix (numeric, ts, mts, zoo, xts)
#' @return A matrix with preserved class and attributes
#' @export
as_matrix_preserve <- function(x) {
  # Input validation
  stopifnot("Input must be one of: numeric, ts, mts, zoo, xts" =
              inherits(x, c("numeric", "ts", "mts", "zoo", "xts", "matrix", "data.frame")))

  # Capture original attributes
  attrs <- attributes(x)
  orig_class <- class(x)

  if (inherits(x, c("zoo", "xts"))) {
      if (requireNamespace("zoo", quietly = TRUE)) {
          orig_index <- zoo::index(x)
          attrs$index <- orig_index
      }
  }

  # Perform coercion
  # For zoo/xts coredata is safer than as.matrix to avoid stripping
  if (inherits(x, c("zoo", "xts"))) {
      if (requireNamespace("zoo", quietly = TRUE)) {
          out <- as.matrix(zoo::coredata(x))
      } else {
          out <- as.matrix(x)
      }
  } else {
      out <- as.matrix(x)
  }

  # Restore attributes except those that conflict with matrix structure
  exclude_attrs <- c("dim", "dimnames", "names")
  attrs_to_restore <- attrs[setdiff(names(attrs), exclude_attrs)]

  if (length(attrs_to_restore) > 0) {
    for (nm in names(attrs_to_restore)) {
      attr(out, nm) <- attrs_to_restore[[nm]]
    }
  }

  class(out) <- orig_class

  return(out)
}
