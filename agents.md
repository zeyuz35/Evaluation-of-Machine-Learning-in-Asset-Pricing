# agents.md

This repository serves dual purposes: it functions simultaneously as (1) an active R package under development, and (2) an active data analysis project containing exploratory scripts, datasets, and simulation studies. These contexts impose different, sometimes conflicting, constraints on code organization and quality standards.

Code progresses through a linear three-stage maturity pipeline:

1. **Analysis**: Exploratory scripts and informal functions developed for immediate insight.
2. **Development**: Formalization of mature analysis code into the package structure (`R/`, `src/`, `tests/`).
3. **Review**: Integration gates verifying that only production-quality, audited code enters the main branch.

**Universal Agents** apply to all stages regardless of context. **Stage-Specific Agents** apply only to code within that phase of the lifecycle. Files in `analysis/`, `scripts/`, or `exploratory/` directories are governed by Analysis agents; files in `R/`, `src/`, and standard package directories are governed by Development and Review agents.

---

## Universal Agents

These standards apply to all R code, C++ code, and configuration files in the repository, regardless of developmental stage.

### agent: Formatting and Style
- Apply `air` formatter with maximum line length of 80 characters
- Enforce one sentence per line, with line breaks occurring only between clauses (typically at punctuation marks)
- Convert all comments to RStudio foldable sections
- Append dashes to headings until reaching approximately 78 characters for visual separation

## agent: Content and Communication Standards 
- Prohibit the use of Unicode emojis in code comments, documentation, commit messages, and markdown files.
- When visual indicators are required (e.g., checkmarks), use markdown-based symbols such as `:heavy_check_mark:` or ASCII alternatives rather than emoji characters.
- Restrict all code comments, documentation strings, and commit messages to ASCII character sets only; avoid non-ASCII characters (including extended Latin, mathematical symbols, or currency symbols) to ensure compatibility across systems with differing locale configurations and encoding defaults.
- Adopt terse, descriptive commit messages following the conventional commits format: `type(scope): description`, where type is one of `feat`, `fix`, `docs`, `style`, `refactor`, `test`, or `chore` (e.g., `feat: add new plotting method`, `fix: correct input validation for ts objects`, `docs: update README with usage examples`).
- Maintain a professional, neutral tone in all code comments, documentation, commit messages, and project communication; avoid informal language, colloquialisms, slang, or humorous asides that may not translate across cultural contexts or could impede professional interpretation.

### agent: Code Modernization
- Replace all magrittr pipes (`%>%`) with native R pipes (`|>`) where syntactically equivalent
- Convert all `for` loops to use `future.apply` framework (`future_lapply`, `future_map`, etc.) to allow user-specified backends
- Rename all single-character iterators (e.g., `i`, `j`, `k`) to double-character equivalents (e.g., `ii`, `jj`, `kk`) for improved searchability and maintainability
- For `lapply` and similar functional iteration, use descriptive iterator names following the pattern `{object_name}_ii` (e.g., `x_vec_ii` when iterating over `x_vec`) to maintain context and searchability

### agent: Console Output and Messaging
- Replace `cat()` calls used for progress reporting or informational messages with `base::message()`
- Convert `cat()` calls indicating recoverable issues to `base::warning()`
- Consolidate patterns of `cat()` followed by `stop()` into single `base::stop()` calls for fatal errors
- Remove explicit newline characters (`\n`) when converting to `message()` or `warning()`, as these functions append newlines automatically
- Utilize `immediate. = TRUE` in `warning()` when warnings are issued inside loops to ensure immediate display rather than deferred batching

### agent: Performance and Readability
- Prioritize readability, maintainability, and clarity over raw speed
- When implementing optimized but obscure code patterns (e.g., `crossprod()` instead of `t(x) %*% x`, or `tcrossprod()` instead of `x %*% t(x)`), add a comment immediately preceding the optimized code showing the equivalent, more readable implementation
- Use vectorized base R operations over manual loops where possible without sacrificing clarity

### agent: Reproducibility and Random Number Generation
- Document RNG stream handling explicitly when using `future.apply` to ensure reproducibility across varying parallel backends (multisession, multicore, etc.)
- Implement explicit `seed` arguments for stochastic functions, utilizing `withr::with_seed()` or similar scoped approaches rather than modifying `.Random.seed` globally
- Ensure proper RNG kind preservation when calling C++ routines via Rcpp that may use C++ standard library random facilities 

### agent: Global State and Side Effects
- Minimize modifications to graphical parameters (`par()`), global `options()`, or working directory (`setwd()`) within function bodies
- Utilize `on.exit()` for cleanup of temporary state changes (e.g., `par(mai=...)`), ensuring cleanup occurs even if the function errors
- Prohibit use of `library()` or `require()` within function definitions; enforce explicit namespacing for all external dependencies

### agent: File System and Path Management
- Replace hardcoded absolute or relative file paths with appropriate abstraction mechanisms (`here::here()` for analysis scripts, `system.file()` for package resources)
- Configure `.gitignore` to exclude `*.Rproj`, `.Rproj.user/`, `*.tar.gz`, `check/` directories, and all binary/compiled files (e.g., `*.o`, `*.so`, `*.dll`, executables) 
- Ensure all binaries are cleaned from the repository working tree and history; do not commit compiled artifacts
- Create temporary files or scripts in `/tmp` (or system temporary directory) rather than the project directory; do not create transient files within the repository tree

---

## Stage 1: Analysis Workflow Agents

These guidelines govern active data analysis scripts, exploratory code, and informal functions prior to formalization.

### agent: Script Organization and Structure
- Organize scripts into workflow-specific directories (e.g., `analysis/`, `scripts/`, `exploratory/`) reflecting pipeline stages (import, clean, model, visualize)
- Utilize `source()` for local function definitions within the analysis context; document dependencies explicitly via comments or project-level environment management (e.g., `renv`)
- Maintain standalone script integrity such that each script executes independently or in defined sequence
- Store simulation templates, drafts, and exploratory code in directories explicitly excluded from package builds via `.Rbuildignore` (e.g., `analysis/`, `dev/`)

### agent: Exploratory Data Management
- Store raw data in `data-raw/` with explicit import scripts documenting provenance
- Store processed datasets in `analysis/results/` or similar with clear metadata; do not place analysis datasets in the package `data/` directory
- Record session information (`sessionInfo()` or `renv::snapshot()`) to ensure reproducibility of analysis environments
- Explicit `rm()` and `gc()` calls are permitted in long-running analysis scripts where manual memory management is required, though reliance on R's automatic garbage collection is generally preferred

### agent: Exploratory Visualization
- Utilize `ggplot2` and specialized visualization packages without restriction to base R graphics
- Prioritize rapid insight generation over formal method standardization
- Conversion to base R `plot()` methods is not required during the analysis phase
- Retain complex `ggplot2` implementations without `autoplot()` registration until formalization into the package structure

### agent: Framework Reuse and Dependency Awareness
- Prioritize established frameworks (e.g., `rsample` for resampling, `recipes` for feature engineering, `tidymodels` for modeling workflows) over custom implementations for standard analytical tasks
- Do not reinvent features that are already well-covered by existing packages unless compelling reasons exist (e.g., performance requirements, simplicity of a bespoke solution, or specific functionality unavailable in existing packages)
- When custom solutions are necessary, document the rationale and the specific gap in existing frameworks

---

## Stage 2: Package Development Agents

These standards apply to the formal R package structure within `R/`, `src/`, `tests/`, and standard package directories.

### agent: Package Structure and Migration
- Move all function definitions from standalone scripts into `R/` directory
- Update all sourcing paths in existing scripts to use `package:::` or `package::` instead of `source()`
- Preserve existing simulation/demo scripts as test templates but do not include in package build
- Organize files logically (e.g., `R/utils.R`, `R/core-functions.R`, `R/cpp-interface.R`)
- Maintain reasonable file lengths by splitting method implementations across multiple files
- Group related methods into separate files using the underscore naming convention
- Place the primary constructor and core class definition in a file named after the class (e.g., `DFM.R`)
- Place specific method categories in appropriately suffixed files (e.g., `DFM_plot.R` for plotting methods, `DFM_predict.R` for prediction methods, `DFM_fitted.R` for extractor methods)
- Ensure that generics defined in `R/generics.R` or similar are collected separately if shared across multiple classes

### agent: Namespace and Dependencies
- Minimize external dependencies; prefer base R implementations unless external packages offer significantly faster/simpler solutions
- Replace all unqualified external function calls with explicit `pkg::function()` syntax
- Replace all internal function calls with explicit `pkg:::function()` syntax where appropriate
- Remove unused imports from DESCRIPTION and NAMESPACE
- Do not add any licenses or modify LICENSE files unless explicitly instructed

### agent: Documentation Standards
- Convert existing pre-function comments (input/output specifications) and post-function comments (examples) into proper roxygen2 documentation
- Ensure all function arguments in roxygen blocks are enclosed in backticks: `\code{arg_name}`
- Format all mathematical expressions using proper LaTeX syntax within roxygen: `\eqn{}` for inline and `\deqn{}` for display equations
- Add `@export` tags for user-facing functions, `@keywords internal` for helper functions
- When features are added/modified, double check to ensure that all documentation is updated accordingly, including examples and parameter descriptions

### agent: C++ Integration (Armadillo)
- When adding RcppArmadillo code, ensure all variables are explicitly initialized
- Use proper C++ namespacing (e.g., `Rcpp::`, `arma::`) consistently
- Maintain a MATLAB/R-like coding style within C++ files for readability, including descriptive variable names, consistent indentation, and comments explaining matrix operations
- Ensure `.cpp` files are placed in `src/` with appropriate `Rcpp::export` attributes

### agent: Class Consistency and Input Handling
- Ensure all functions handle inputs of classes: `numeric`, `ts`, `mts`, `zoo`, `xts`, and occasionally `tsibble`/`data.frame`
- The output class must match the input class exactly, preserving time indices (accounting for appropriate lags), column names, and attributes specific to the class
- For computationally expensive operations on non-numeric inputs, extract core data using `coredata()`, process numerically, then re-attach original indices and attributes to coerce back to the original class
- Prior to `coredata()` extraction or any coercion that strips attributes, explicitly capture and store all relevant attributes (particularly those related to scaling and transformations) in a temporary object
- After numerical processing, restore all captured attributes to the output object before returning, ensuring that scaling parameters, transformation flags, and other metadata remain intact for subsequent inverse operations
- Handle missing data patterns appropriately across all supported input types

### agent: Formal Plotting Methods
- Convert simple `ggplot2` implementations to base R `plot()` methods where the visualization is straightforward (e.g., single time series, basic scatter)
- Register existing `ggplot2` implementations as `autoplot()` methods (S3) from the `ggplot2` namespace
- For complex visualizations requiring `ggplot2` aesthetics, retain as `autoplot()` methods without base R equivalents
- Ensure plot methods respect the input object class (e.g., preserve time indices for `ts`/`xts` objects)

### agent: Testing Framework
- Implement tests using `testthat` framework in `tests/testthat/`
- Structure test files to load the built package (e.g., `library(pkgname)`) rather than sourcing individual files
- Convert existing simulation studies and demo scripts into formal test cases, particularly for input/output class consistency, numerical accuracy against known benchmarks, and edge cases (missing data, single observations)
- Test comprehensively across all supported input classes (`numeric`, `ts`, `mts`, `zoo`, `xts`)

### agent: Package Data and Configuration
- Migrate hardcoded datasets to `data/` (exported) or `R/sysdata.rda` (internal) via `usethis::use_data()`, ensuring compliance with lazy-loading mechanisms
- Replace global configuration variables and hardcoded constants with `options()` infrastructure, documented under `?pkgname-options` or similar help topic
- Remove all explicit `rm()` and `gc()` calls from function bodies, as these interfere with R's automatic memory management and garbage collection 

### agent: Input Validation and Error Handling
- Implement robust input validation using `checkmate`, `assertthat`, or base R `stopifnot()` with informative, contextual error messages that preserve the class information of time series objects
- Ensure validation occurs prior to `coredata()` extraction to enable meaningful error messages referencing the original object structure
- Specify `call. = FALSE` in `warning()` and `stop()` for internal validation, to avoid exposing internal function names to end users
- Utilize `packageStartupMessage()` within `.onAttach()` for package load notifications, rather than `cat()` or standard `message()`

### agent: Documentation Architecture
- Create package-level documentation in `R/pkgname-package.R` using the `_PACKAGE` sentinel and `@keywords internal` for comprehensive package overview
- Convert extended simulation examples into formal vignettes in `vignettes/`, ensuring `VignetteBuilder: knitr` in DESCRIPTION
- Maintain `README.Rmd` (not merely `.md`) for dynamic badge generation and executable examples via `reprex` standards

### agent: S3 Method Consistency and Dispatch
- Audit S3 generic functions (`plot`, `predict`, `fitted`, `residuals`, etc.) for consistent argument signatures across all methods
- Ensure `autoplot` methods properly import `ggplot2::autoplot` via roxygen `@importFrom ggplot2 autoplot` and re-export if the package exposes its own generics
- Verify that `UseMethod()` dispatch occurs after validation but before any class-stripping operations like `as.matrix()` or `unclass()`

---

## Stage 3: Review and Integration Agents

These standards apply to pull requests, continuous integration, and pre-merge verification gates.

### agent: Build Verification
- All code changes (excluding documentation-only or README updates) must trigger a package build (`R CMD build` or `devtools::build()`)
- Run the complete test suite (`devtools::test()` or `R CMD check`) and ensure all tests pass before submitting changes
- Verify no new warnings or notes are introduced during `R CMD check`

### agent: Continuous Integration and Quality Assurance
- Configure GitHub Actions workflows for `R CMD check` across Linux, macOS, and Windows with multiple R versions (release, devel, oldrel)
- Integrate code coverage tracking via `covr` and codecov.io, excluding C++ branches from coverage metrics where appropriate
- Add `lintr` static analysis to CI pipeline to enforce style consistency alongside `air` formatting 

### agent: C++ Build Configuration
- Create `src/Makevars` and `src/Makevars.win` with appropriate `PKG_CPPFLAGS` and `PKG_LIBS` flags for RcppArmadillo linking, including `-fopenmp` where applicable
- Specify `SystemRequirements: C++17` (or appropriate standard) in DESCRIPTION and ensure `LinkingTo: Rcpp, RcppArmadillo` is present
- Implement `Rcpp::Rcpp.plugin.maker()` registration if utilizing custom Rcpp attributes or plugins 

### agent: Artifact Hygiene
- Verify that no analysis-specific files (scripts, datasets, temporary outputs) have migrated into package directories (`R/`, `src/`, `tests/`, `man/`, `inst/`)
- Validate that `.Rbuildignore` excludes all development artifacts (`analysis/`, `scripts/`, `*.Rproj`, build artifacts)
- Confirm no compiled binaries (`.o`, `.so`, `.dll`) exist in the git history or working tree; enforce `src/*.o` and `src/*.so` in `.gitignore`
- Ensure temporary files are never created within the repository tree during CI or local builds

### agent: Security Scanning
- Detect and remove hardcoded credentials, API keys, or database connection strings in migrated code
- Prohibit use of `system()`, `shell()`, or `eval(parse())` with unsanitized user inputs
- Verify that `serialize()`/`unserialize()` or `readRDS()` on untrusted sources is not introduced during analysis-to-package migration
- Ensure file paths in examples and vignettes do not reference absolute paths or sensitive system locations

### agent: Namespace Pollution Check
- Verify that analysis-specific dependencies (e.g., visualization packages used only for exploration) are not added to DESCRIPTION unless formally required by package methods
- Confirm that `Suggests` versus `Imports` distinction correctly reflects runtime requirements versus development or analysis needs
- Ensure no orphaned `library()` calls remain in migrated code from analysis scripts
