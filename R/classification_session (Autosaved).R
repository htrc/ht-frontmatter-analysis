library(caret)
library(e1071)

x = read.csv("./data/training_data.csv", sep=" ", header=T)
x = x[,-c(1,2)]

set.seed(7)
train_index = createDataPartition(x$target, p=0.8, list=FALSE)
x.train = x[train_index,]
x.test  = x[-train_index,]

control = trainControl(method="cv", number=10)
metric = "Accuracy"

fit.lda = train(target~., data=x.train, method="lda", metric=metric, trControl=control)
fit.cart = train(target~., data=x.train, method="rpart", metric=metric, trControl=control) 
fit.knn = train(target~., data=x.train, method="knn", metric=metric, trControl=control) 
fit.lr  = train(target~., data=x.train, method="glm", family="binomial", metric=metric, trControl=control)
fit.svm = train(target~., data=x.train, method="svmRadial", metric=metric, trControl=control) 
fit.rf = train(target~., data=x.train, method="rf", metric=metric, trControl=control)


results = resamples(list(lda=fit.lda, cart=fit.cart, lr=fit.lr, knn=fit.knn, svm=fit.svm, rf=fit.rf)) 

summary(results)
dotplot(results)

predictions = predict(fit.rf, x.test)
confusionMatrix(predictions, x.test$target)


n = c(100, 250, 500, 1000, 2500, 5000, 7500, 10000, 15000, 24214)
a.lr  = vector()
a.svm = vector()
a.rf  = vector()

for(i in 1:length(n)) {
	print(i)
	x.sample = x.train[sample(nrow(x.train), n[i]),]
	fit.lr  = train(target~., data=x.sample, method="glm", family="binomial", metric=metric, trControl=control)
	fit.svm = train(target~., data=x.sample, method="svmRadial", metric=metric, trControl=control)
	fit.rf  = train(target~., data=x.sample, method="rf", metric=metric, trControl=control)
	predictions = predict(fit.lr, x.test)
	cm = confusionMatrix(predictions, x.test$target)
	a.lr[i] = as.numeric(cm[3]$overall[1])
	
	predictions = predict(fit.svm, x.test)
	cm = confusionMatrix(predictions, x.test$target)
	a.svm[i] = as.numeric(cm[3]$overall[1])
	
	predictions = predict(fit.rf, x.test)
	cm = confusionMatrix(predictions, x.test$target)
	a.rf[i] = as.numeric(cm[3]$overall[1])
	
	
}
plot(cbind(n,a.rf), ylim=c(0.88, 0.95), type="b", xlab="Training Instances (# Pages)", ylab="Model Accuracy")
points(cbind(n,a.svm), type="b", col="red")
points(cbind(n,a.lr),  type="b", col="blue")
legend(20000, 0.91, legend=c("RF", "SVM", "LR"), col=c("black", "red", "blue"), pch=c(1,1,1), cex=0.75 )

filterControl = sbfControl(functions=rfSBF, method="repeatedCV", repeats=5)
set.seed(7)
fbWithFiler = sbf(x[,1:11], x$target, sbfControl=filterControl)


