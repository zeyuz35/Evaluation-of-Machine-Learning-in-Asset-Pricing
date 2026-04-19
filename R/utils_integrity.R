as_matrix_preserve <- function(x, ...) {
  if (inherits(x, c("ts", "xts", "zoo"))) {
    attrs <- attributes(x)
    mat <- as.matrix(x, ...)
    preserved_attrs <- setdiff(names(attrs), c("dim", "dimnames"))
    for (attr_name in preserved_attrs) {
      attr(mat, attr_name) <- attrs[[attr_name]]
    }
    return(mat)
  }
  return(as.matrix(x, ...))
}
