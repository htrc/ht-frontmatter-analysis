The prelim experiment on this project providing a fairly good result, However there are some take aways and considerations to continue moving forward on this project

-   The preliminary experiment uses statistical values to achieve the result. This statistic values that might fairly easy to be derived from extracted feature (Need to check the extracted feature set if it contains these values or not)

![](https://storage.googleapis.com/slite-api-files-production/files/4802650e-e258-4977-9cb1-6a55dbe58d44/image.png)

-   The root elements/variables are:
    -   seq
    -   token_count
    -   line_count
    -   empty_line_count
    -   cap_alpha_seq
    -   pct_begin_char_caps
    -   pct_end_char_numeric
    -   num_roman_numerals
    -   pct_all_caps

Result on reproducing the feature values using Python:

![](https://storage.googleapis.com/slite-api-files-production/files/8fecf07c-2edc-46ae-86b2-7966925410a0/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/422d09b4-32ee-4942-85d7-c667a0bad6f9/image.png)

-   Besides those elements above, there are some variables that are derived from the root elements such as token_count_normalized and token_count. Although this variables are most highly have high correlation one to the others (because they are having functional dependencies), these variables were used on the prelim experiment. 

Therefore, another next step that we can do is providing some statistical analysis to determine which features are the most representative to achieve our goal so we can eliminate  such features that we don't need.

-   We can get the Extracted Features using Python library provided by HTRC that will be useful to reproduce the input data on this statistical model. Example code in the Jupyter Notebook:

https://github.com/nikolausn/ht-frontmatter-analysis/blob/master/Jupyter/HTRC_FrontMatter_Analysis.ipynb

-   Although the term frequency model that we presented as an alternative works better, it is a High Cost model, means it takes a lot of feature space (words as feature) to represent the word/terms frequency. There are some drawbacks on this method in which it will not one model might not be generalize enough to represent every book we have on the collection. One example is, if we have a new term on different collection set, it will not be represented in the model and we need to retrain the model.

This model might not be useful in the end, however it still can be useful to use as the baseline to understand how accurate the statistical model compare to this word freq model (or we might use embedding + lstm model as well).
