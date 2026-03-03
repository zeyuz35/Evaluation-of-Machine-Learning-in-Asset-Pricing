library(testthat)

# Use absolute path resolving or assume standard package load for tests
# In local development we can just source directly as we know we're running from root
source(file.path(Sys.getenv("PWD", "."), "R/utils_integrity.R"))

test_that("as_matrix_preserve keeps attributes", {
  x <- scale(mtcars)
  expect_true(!is.null(attr(x, "scaled:center")))
  expect_true(!is.null(attr(x, "scaled:scale")))

  m <- as_matrix_preserve(x)
  expect_true(!is.null(attr(m, "scaled:center")))
  expect_true(!is.null(attr(m, "scaled:scale")))
})

test_that("as_matrix_preserve errors on NULL", {
  expect_error(as_matrix_preserve(NULL))
})

test_that("as_matrix_preserve preserves tsp for ts", {
  x <- ts(1:10, start = c(2020, 1), frequency = 12)
  attr(x, "custom_attr") <- "value"

  m <- as_matrix_preserve(x)
  expect_true(!is.null(attr(m, "custom_attr")))
  expect_equal(attr(m, "custom_attr"), "value")
  expect_true(!is.null(attr(m, "tsp")))
})
