#' Safely convert objects to matrix while preserving critical attributes
#'
#' @param x An object to convert to matrix (e.g., ts, mts, zoo, xts, data.frame)
#' @param ... Additional arguments passed to base::as.matrix
#' @return A matrix with original attributes restored
as_matrix_preserve <- function(x, ...) {
  if (is.null(x)) {
    stop("Input cannot be NULL")
  }

  # Preserve original attributes
  orig_attrs <- attributes(x)

  # Extract numeric matrix based on class
  if (inherits(x, c("zoo", "xts"))) {
    if (!requireNamespace("zoo", quietly = TRUE)) {
      stop("Package 'zoo' is required for zoo/xts objects.")
    }
    res <- zoo::coredata(x)
  } else {
    res <- base::as.matrix(x, ...)
  }

  # Restore attributes that don't conflict with basic matrix structure
  # We preserve scaling attributes, transformation metadata, and custom properties
  if (!is.null(orig_attrs)) {
    attrs_to_keep <- orig_attrs[setdiff(names(orig_attrs), c("dim", "dimnames", "class", "index", "tsp", "names", "row.names"))]

    if (length(attrs_to_keep) > 0) {
      for (attr_name in names(attrs_to_keep)) {
        attr(res, attr_name) <- attrs_to_keep[[attr_name]]
      }
    }
  }

  return(res)
}
