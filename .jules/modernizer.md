## Modernizer Journal
## 2026-04-01 - Modernized Pipes and Loop Iterators in Rmd files

**Learning:** When performing string replacements for native pipe |> or iterator renames in Rmd files, it's essential to verify the changes by rendering the file using rmarkdown::render(). Installing necessary dependencies temporarily allows proper testing and confirmation that logic and visuals are identical. Using sed to substitute specific inner-loop variables (e.g. lnsig2[i-1]) is effective.

**Action:** Always follow up text substitutions in Rmd with a rendering test to guarantee no regressions in syntax or output.
