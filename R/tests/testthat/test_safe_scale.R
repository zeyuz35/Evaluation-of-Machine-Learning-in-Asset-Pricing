library(testthat)

source("../../utils/safe_scale.R")

test_that("safe_scale preserves ts class and attributes", {
  test_ts <- ts(1:10, frequency = 4, start = c(1959, 2))
  scaled_ts <- safe_scale(test_ts)

  # Check class
  expect_true(inherits(scaled_ts, "ts"))

  # Check standard attributes
  expect_equal(tsp(scaled_ts), tsp(test_ts))

  # Check scale attributes
  expect_true(!is.null(attr(scaled_ts, "scaled:center")))
  expect_true(!is.null(attr(scaled_ts, "scaled:scale")))
})

test_that("safe_scale preserves xts class and attributes", {
  if (requireNamespace("xts", quietly = TRUE)) {
    test_xts <- xts::xts(1:10, order.by = as.Date("2000-01-01") + 0:9)
    scaled_xts <- safe_scale(test_xts)

    expect_true(inherits(scaled_xts, "xts"))

    expect_true(!is.null(attr(scaled_xts, "scaled:center")))
    expect_true(!is.null(attr(scaled_xts, "scaled:scale")))
  }
})
