A work has been done in the past. Based on the Miles Efron project on the Github

https://github.com/milesefron/ht-frontmatter-analysis , I tried to reproduce the code using R since the original code was written on R.

However, I noticed several problems when I tried to reproduce the experiment, as follow:

-   Some libraries that were used on previous experiment has gone several changes (revision). Therefore, if we run the code directly we will experience some errors, especially on the training the models part. 

**Before**:

![](https://storage.googleapis.com/slite-api-files-production/files/7a76b803-b41b-4d25-931a-fc69c47ca91f/image.png)

The trainControl part with method="none" is not working on knn, svm, and rf model mainly because they changed implementation to avoid confusion between method in the train function and the traincontrol function. That will produce this error:

![](https://storage.googleapis.com/slite-api-files-production/files/493304de-1881-4b2e-baf6-1008d10d3fcc/image.png)

**After:**

For this case I changed the trainControl code to follow the new rule set as follow, removing the method parameter from the trainControl function

![](https://storage.googleapis.com/slite-api-files-production/files/a928de65-3ce3-43c9-b53b-a8b5ddbda18c/image.png)

And now the code works normally

-   After fixing the code, I experienced that some model training run really slow that can takes more than half hour to finish especially for the knn, and svm model. Some possibilities of why it runs slow are:
    -   My hardware is out of date: I used Macbook pro 2013 with I7 (third gen) processor and 16 GB memory. For most of my use case, this gadget can run well, so I don't think hardware is the issue here
    -   R is just slow: Maybe the R just running slow in my computer
    -   The method (training model function) that the packages provided are not properly tuned in which the computational complexity of the model is just very costly. 
-   Besides that, I noticed that the svm model failed to reach convergence when I executed the svm training code. This might be happened because of the new code on the new release. However, if this is doesn't happen previously on the Miles Efron work, then this might impact on a serious issue on the result

![](https://storage.googleapis.com/slite-api-files-production/files/bc3abbf2-da29-4a6f-8ece-364964111fc2/image.png)

With these limitations that we found on reproducing the work, I still can reproduce some part of the work and the code and produce "somehow" similar results with the one shown in the Github.

**From the Github Markdown:**

![](https://storage.googleapis.com/slite-api-files-production/files/e17d5ba4-31c1-437d-946b-46da4cfde2dd/image.png)

**Result from our reproduce experiment:**

![](https://storage.googleapis.com/slite-api-files-production/files/8f75450b-27f6-4e1f-aec4-c74ea6df8eee/image.png)

As we can see on the figure above, although in general the method produces the same pattern which RF works better compare to the SVM and followed by LR in terms of the model accuracy, it does not exactly produce the same plot. Besides that, we can see a super drop in the end that might be a product of the error produced by the SVM function we experienced when executing the code.

Facing this issue on the model performance in R, I tried to replicate the same experiment in different environment: Python. in the next story, I present the same approach but we run it on  the Python using Python packages and see if we produce a same result or not.
