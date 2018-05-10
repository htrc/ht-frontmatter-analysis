library(caret)
library(e1071)
source("./R/partition_data.r")

x = read.csv("./data/java_training_data.csv", sep=" ", header=T, row.names=NULL)

split = partition_data(x);
x.train = split[['train']]
x.test  = split[['test']]
x.train = x.train[,-1]
x.test  = x.test[,-1]

#control = trainControl(method="cv", number=10, classProbs=TRUE)
control = trainControl(method="none", classProbs=TRUE)
metric = "Accuracy"

#fit.first  = train(target~., data=x.train, method="glm", #family="binomial", metric=metric, trControl=control)

fit.first  = train(target~., data=x.train, method="rf",  metric=metric, trControl=control)

predictions.test = predict(fit.first, x.test)
confusionMatrix(predictions.test, x.test$target)

predictions.train = predict(fit.first, type="prob")
predictions.test = predict(fit.first, type="prob", newdata=x.test)
open.ind = which(names(predictions.train)=='open')


expand_features = function(x, predictions) {
	open.ind = which(names(predictions)=='open')
	bwd = rep(1,nrow(x))
	bwd[2:nrow(x)] = predictions[,open.ind][1:(nrow(x)-1)]
	fwd = rep(1,nrow(x))
	fwd[1:(nrow(x)-1)] = predictions[,open.ind][2:(nrow(x))]
	bwd[1] = predictions[,open.ind][1]
	fwd[nrow(x)] = predictions[,open.ind][nrow(x)]
	
	return(cbind(x, bwd, fwd))
	#return(cbind(x, bwd))
}



# now we make our second model

# first, expand our training data to include context
x.train = expand_features(x.train, predictions.train)

#interaction = x.train$fwd * x.train$bwd
#x.train = cbind(x.train, interaction)

# now fit the model
#fit.meta  = train(target~., data=x.train, method="glm", #family="binomial", metric=metric, trControl=control)

fit.meta  = train(target~., data=x.test, method="rf",  metric=metric, trControl=control)


# expand our test matrix
x.test = expand_features(x.test, predictions.test)

#interaction = x.test$fwd * x.test$bwd
#x.test = cbind(x.test, interaction)


predictions.meta = predict(fit.meta, x.test)
confusionMatrix(predictions.meta, x.test$target)
