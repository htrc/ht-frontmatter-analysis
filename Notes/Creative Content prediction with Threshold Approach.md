From our previous approach, we can see that random forest model with statistical features works well predicting creative content and non creative content data.

Now we want to bring this random forest model to the next level by analyzing the threshold for the random forest prediction so we can use better prediction for creative content which is more important than predicting the factual content. On the other term we want to increase the recall for the creative content prediction.

First I use cross validation to build models based on the training set and evaluate it over the validation set. From this step we can get the best model that generalize better on the cross validated set

![](https://storage.googleapis.com/slite-api-files-production/files/0814762c-aac8-4a20-a50a-d4d98370e0df/image.png)

We get the best model from the cross validation process and look up at the distribution of false negative prediction on the creative content prediction.

![](https://storage.googleapis.com/slite-api-files-production/files/f379938a-2a5f-4db0-a7c3-b04cdb1d0293/image.png)

As we can see from the plot, the probability of the creative content when we are detecting false prediction is quite vary with range 0 to 0.48. Further we compute the statistics, mean and max for the output probability of creative content

![](https://storage.googleapis.com/slite-api-files-production/files/451c576a-c620-4b4d-b2e9-87a145cd0ca4/image.png)

As we can see from the statistics, the, average probability for the error is 0.18. This where we decide the threshold for detecting creative content. Based on the mean statistic result, we decided to use 0.2 as the threshold for the creative content prediction

We evaluate our choose in the testing set and with this method, we can achieve 94% recall (accuracy) for creative content prediction. However, this threshold method eventually will hurt the factual content prediction as we can see 24% of the testing set are predicted as creative content. However, again depends on our goal, we care the most of creative content.

![](https://storage.googleapis.com/slite-api-files-production/files/ae2c9f91-d3e3-42f3-8daa-9cb94193cbd3/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/42856df5-7ffe-49fb-94c3-2c470519bb61/image.png)

Using sliding window to detect the creative content (importance of the order and previous pages) we used the same approach with threshold computation

I augmented the dataset to attach the statistics of two previous pages on the dataframe (all data) and split it into training and testing later

![](https://storage.googleapis.com/slite-api-files-production/files/79aee5d5-aa1a-4077-b6b7-575fefdf8fcd/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/f4f4e241-e789-44fe-8997-4133c87e9707/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/17b0310b-6f91-45f2-a59e-5aa053403523/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/fe5745a9-f790-44e2-8132-b2d475212049/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/e4f5488c-0e15-4d1b-9eab-17e4607bb05b/image.png)

![](https://storage.googleapis.com/slite-api-files-production/files/73264c9f-295f-45a5-ad14-41c016bd1cd0/image.png)

With sliding window approach, we can get 96% recall which is 2% better than our standard random forest model
