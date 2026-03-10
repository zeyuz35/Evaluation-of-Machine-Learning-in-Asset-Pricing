library(testthat)

# Note: The codebase doesn't have a fully formal package structure right now,
# but we are sourcing the function manually for these tests.
source("../../R/utils_integrity.R")

test_that("as_matrix_preserve works correctly", {
  # 1. Test numeric matrix with custom attributes
  m <- matrix(1:4, 2, 2)
  attr(m, "custom_attr") <- "test"
  m_out <- as_matrix_preserve(m)
  expect_equal(attr(m_out, "custom_attr"), "test")

  # 2. Test xts with scale
  if (requireNamespace("xts", quietly = TRUE)) {
    data <- matrix(rnorm(100), ncol=10)
    data_scaled <- scale(data)
    dates <- as.Date("2023-01-01") + 0:9
    x_scaled <- xts::xts(data_scaled, order.by=dates)

    x_out <- as_matrix_preserve(x_scaled)
    expect_true(!is.null(attr(x_out, "scaled:center")))
    expect_true(!is.null(attr(x_out, "scaled:scale")))
    expect_true(!is.null(attr(x_out, "index")))
  }

  # 3. Test ts object with scale
  ts_data <- ts(scale(matrix(rnorm(100), ncol=10)), start = c(2020, 1), frequency = 12)
  ts_out <- as_matrix_preserve(ts_data)
  expect_true(!is.null(attr(ts_out, "scaled:center")))
  expect_true(!is.null(attr(ts_out, "scaled:scale")))
})
