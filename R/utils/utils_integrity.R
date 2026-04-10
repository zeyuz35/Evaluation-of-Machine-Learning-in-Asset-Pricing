#' Coerce to matrix while preserving time-series attributes
#'
#' Base R's \code{as.matrix()} strips class and attributes from time-series objects
#' like \code{ts}, \code{xts}, and \code{zoo}. This function preserves them.
#'
#' @param x An object to coerce to matrix.
#' @param ... Additional arguments passed to \code{as.matrix()}.
#' @return A matrix with the preserved attributes of the input object.
#' @export
as_matrix_preserve <- function(x, ...) {
  stopifnot("Input must not be NULL" = !is.null(x))

  if (inherits(x, c("zoo", "ts", "xts", "mts"))) {
    attrs <- attributes(x)
    mat <- as.matrix(x, ...)

    # We want to preserve specific attributes like class, scaling, time indices,
    # but dim/dimnames might change when coercing to matrix.
    # Therefore, we only restore non-structural ones, or merge them properly.
    attributes(mat) <- utils::modifyList(attributes(mat),
                                  attrs[setdiff(names(attrs),
                                                c("dim", "dimnames"))])
    return(mat)
  }
  return(as.matrix(x, ...))
}
