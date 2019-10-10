I used the dataset shared by colleague Kristina Eden: https://www.hathitrust.org/files/FrontmatterData.tsv

Firstly the dataset is contains different collection than the front-matter analysis. Therefore, we need to get the extracted feature from the given collection and build the same statistical features we used on the front-matter analysis. Next, the target class is also different, instead of detecting front page and main content, we are now detecting the content, predicting whether it is a creative content or main content. To do this, I developed a new workflow on downloading, and providing the statistical values for each document.

![](https://storage.googleapis.com/slite-api-files-production/files/c884a72b-bb51-4d15-8d8c-646df8dcb4ef/image.png)

parameter t is the page from the extracted feature class.

From the tsv file, I noticed that there are two type of target variables that can be inferred:

-   Factual vs Creative: we call this main class

![](https://storage.googleapis.com/slite-api-files-production/files/f0b0a67c-dafa-4272-9b47-2d61da826646/image.png)

-   Detail type, which contain more detail explanation about the class (subclass). 

![](https://storage.googleapis.com/slite-api-files-production/files/17b5d245-3028-470f-8d89-eac654007bc9/image.png)

# Main class model analysis

The first analysis is done on the main class. Here we are comparing model using the Random Forest and Logistic Regression as those model perform the best on our front matter detection.

![](https://storage.googleapis.com/slite-api-files-production/files/6d6aec74-b445-4d58-8ffe-a3fd0724e5be/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/3436494f-3381-4924-b17f-142bd761f90d/image.png)

Based on our model observation above, the Random Forest model performs well with 85% accuracy and can work on almost all the class. The model have low accuracy predicting the mixed content but overall this model works with a fairly good performance. Compare to the Random Forest model, the Logistic Regression has lower accuracy with 83% and failed to distinguish mixed content. As conclusion, using a statistical measurement derived from extracted feature can help us predicting the main class and we can achieve up to 85% accuracy on the testing set.

# Sub-class model analysis

After generating a model for the Main class, we did the same thing on the subclass using more predicted values.

![](https://storage.googleapis.com/slite-api-files-production/files/baf26cf0-0659-41e1-8d7b-134a3fc94fe5/image.png)

As we can see from the result on the sub class model with random forest predictor, the model can only gain 70% accuracy and has trouble predicting some of the classes. Furthermore, the confusion can be seen on this metric

![](https://storage.googleapis.com/slite-api-files-production/files/88f7443f-fb2e-47fc-a171-77c07f798801/image.png)

The random forest with statistical feature cannot event predict the epigraph and has the most confusion on detecting dedication text. Furthemore looking at the confusion matrix above we can see that title vs pub_info, main_text vs pref_text, and ad vs image has the most confused cluster.

Looking at this particular result, I tried second approach using the feature words as our training data and perform sgd classifier on the term frequency

![](https://storage.googleapis.com/slite-api-files-production/files/ec842b4a-08b6-4fd6-a186-56144955c79d/image.png)

Although the accuracy performance is quite the same, the precision between classes are performing really well compare to the statistic approach. Furthermore, the detail can be seen on the confusion matrix below

![](https://storage.googleapis.com/slite-api-files-production/files/dd2d4f38-94c4-4f3e-b891-4bba71bf6bcb/image.png)

The term frequency variable feature it produces more fair distribution on the prediction and can predict some of the epigraph contents correctly. Besides that, the poem, appendix, ad, list, and main text result on significant improvement compared to using only statistical values.
