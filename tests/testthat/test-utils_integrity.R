library(testthat)
source("../../R/utils_integrity.R")

test_that("as_matrix_preserve preserves attributes but drops class for xts", {
  library(xts)
  x <- xts(matrix(1:10, ncol=2), order.by=Sys.Date() + 1:5)
  attr(x, "scaled:center") <- c(0, 0)
  attr(x, "scaled:scale") <- c(1, 1)

  y <- as_matrix_preserve(x)

  expect_false(inherits(y, "xts"))
  expect_true(is.matrix(y))
  expect_equal(attr(y, "scaled:center"), c(0, 0))
  expect_equal(attr(y, "scaled:scale"), c(1, 1))
  expect_equal(attr(y, "index"), attr(x, "index"))
})

test_that("as_matrix_preserve preserves attributes but drops class for ts", {
  x <- ts(1:10, frequency=4, start=c(1959, 2))
  attr(x, "scaled:center") <- 0
  attr(x, "scaled:scale") <- 1

  y <- as_matrix_preserve(x)

  expect_false(inherits(y, "ts"))
  expect_true(is.matrix(y))
  expect_equal(attr(y, "scaled:center"), 0)
  expect_equal(attr(y, "scaled:scale"), 1)
  expect_equal(attr(y, "tsp"), attr(x, "tsp"))
})

test_that("as_matrix_preserve preserves attributes but drops class for zoo", {
  library(zoo)
  x <- zoo(matrix(1:10, ncol=2), order.by=1:5)
  attr(x, "scaled:center") <- c(0, 0)

  y <- as_matrix_preserve(x)

  expect_false(inherits(y, "zoo"))
  expect_true(is.matrix(y))
  expect_equal(attr(y, "scaled:center"), c(0, 0))
  expect_equal(attr(y, "index"), attr(x, "index"))
})

test_that("as_matrix_preserve preserves generic attributes on numeric matrices", {
  x <- matrix(1:10, ncol=2)
  attr(x, "custom_attr") <- "test"

  y <- as_matrix_preserve(x)

  expect_equal(attr(y, "custom_attr"), "test")
  expect_true(is.matrix(y))
})
