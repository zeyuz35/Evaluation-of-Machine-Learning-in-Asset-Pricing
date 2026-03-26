1. **Target**: `R/AWS/sagemaker_demo.R`
2. **Current Pattern**: Several magrittr pipes (`%>%`) are used for simple variable mutations/filterings:
```R
abalone <- abalone %>%
  filter(height != 0)

abalone <- abalone %>%
  mutate(female = as.integer(ifelse(sex == 'F', 1, 0)),
         male = as.integer(ifelse(sex == 'M', 1, 0)),
         infant = as.integer(ifelse(sex == 'I', 1, 0))) %>%
  select(-sex)
abalone <- abalone %>%
  select(rings:infant, length:shell_weight)
head(abalone)

## Train/test split

abalone_train <- abalone %>%
  sample_frac(size = 0.7)
abalone <- anti_join(abalone, abalone_train)
abalone_test <- abalone %>%
  sample_frac(size = 0.5)
```
3. **Proposed Modernization**: Convert the `%>%` pipes to `|>` native pipes. Ensure that the right side of the pipe uses proper function calls. This perfectly fits the "Modernizer" persona as an under-50 lines refactor reducing dependencies and improving maintainability.
4. **Benefit**: Eliminates the need for the `magrittr` dependency in this context and standardizes pipe usage on the native pipe (`|>` R >= 4.1.0).
5. **Testing**: `devtools::test()` or simply parse and check that it's valid code, then submit the PR.
6. **Complete pre-commit steps to ensure proper testing, verification, review, and reflection are done.**
