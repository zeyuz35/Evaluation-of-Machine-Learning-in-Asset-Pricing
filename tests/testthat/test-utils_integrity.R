library(zoo)
library(xts)

source(here::here("R", "utils", "utils_integrity.R"))

test_that("as_matrix_preserve keeps ts attributes", {
  test_ts <- ts(1:10, start = c(2000, 1), frequency = 12)
  attr(test_ts, "scaled:scale") <- 2

  mat <- as_matrix_preserve(test_ts)
  expect_true(!is.null(attr(mat, "tsp")))
  expect_true(!is.null(attr(mat, "scaled:scale")))
})

test_that("as_matrix_preserve keeps xts attributes", {
  test_xts <- xts(1:10, order.by = as.Date("2000-01-01") + 0:9)
  attr(test_xts, "scaled:scale") <- 3

  mat <- as_matrix_preserve(test_xts)
  expect_true(!is.null(attr(mat, "index")))
  expect_true(!is.null(attr(mat, "scaled:scale")))
})
