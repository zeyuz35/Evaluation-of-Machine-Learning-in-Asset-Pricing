Monash Honours Thesis Rmarkdown Template
=========================================

This repository provides a template for a Monash University Honours thesis using Rmarkdown with the bookdown package. It is designed for Honours students in the Department of Econometrics and Business Statistics, but can be modified for other departments and other universities as required. It is based on a [similar template for PhD students](https://github.com/robjhyndman/MonashThesis).

## Requirements

To set up the software, you will need to install the `bookdown` package and its dependencies as follows:

```r
install.packages('bookdown')
```

You will also need LaTeX installed. If you don't already have LaTeX, one convenient approach is to install it via R:

```r
install.packages('tinytex')
tinytex::install_tinytex()
```
