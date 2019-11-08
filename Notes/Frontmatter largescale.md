Here I present an approach (proof of concept) that might be useful to use spark capabilities to enhance the model one at a time using Human in the loop strategy and iterative learning process.

The purpose of training the model is not for determining the final prediction rather to provide a suggestion that can help manual coders do their work by focusing on the creative content prediction class.

The former workflow for the machine learning modeling can be seen as follow:

 

![](https://storage.googleapis.com/slite-api-files-production/files/9668df84-3439-476e-a402-cd2eb63d819f/null)

And here is the propose workflow with human in the loop and iterative dataset collection / enhancement :

![](https://storage.googleapis.com/slite-api-files-production/files/b20faff3-3b80-4498-9b55-1adaab4163fe/null)

As we can see in this proposed workflow, our purpose is to strategically collect sample from the misclassified prediction to fulfill our label_collection dataset. In the end product, based on the executive decision, we can use either only the labeled collection or the model directly to support the front-matter display feature.

Because I don't have spark cluster on my reach, I prototyped this method using Databricks. 

<https://community.cloud.databricks.com/>

Databricks is a manage service company developed by the founder of spark to provide the best end user experience for the user to jump in to spark cluster and analytics easily without worrying to much about the infrastructure. In this research, I used pyspark as my programming language of choose because I am more familiar with python. But I believe this will be pretty much transformable to scala.

The Databricks notebook is accessible on Github:

<https://github.com/htrc/ht-frontmatter-analysis/blob/master/Jupyter/frontmatter-scale-onspark.ipynb>

and published on databricks domain

https://databricks-prod-cloudfront.cloud.databricks.com/public/4027ec902e239c93eaaa8714f173bcfc/6536095386397355/433444548221712/1631695512458656/latest.html

Another case to check on

Volumes that can't be found using HTRC API:

| **vol_id**          |
| :------------------ |
| uc1.l0075873877     |
| txu.059173014313723 |
| umn.31951d010311165 |
| uiug.30112121939497 |
| nyp.33433082868591  |
| uc1.l0064487499     |
| pst.000047198050    |
| osu.32435075731331  |
| uc1.b000580094      |
| umn.31951p01038978c |
| uc1.x36249          |
| uc1.x78904          |
| osu.32435030731681  |
| uc1.x33356          |
| umn.31951p00589914c |
| uiug.30112025675866 |
| umn.31951d00569039k |
| umn.31951d00185986b |
| osu.32435030710412  |
| uc1.x58537          |
| umn.319510006422813 |
| uiug.30112049744227 |
| uc1.c049367576      |
| osu.32435005569124  |
| osu.32435064221757  |
