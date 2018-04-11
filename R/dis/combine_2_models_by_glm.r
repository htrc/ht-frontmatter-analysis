library(caret)
library(e1071)

source("./R/partition_data.r")
source("./R/probs2seq.r")

## data wrangling
x = read.csv("./data/java_training_data.csv", sep=" ", header=T, row.names=NULL)

set.seed(7)
split = partition_data(x);
x.train = split[['train']]
x.test  = split[['test']]
x.train = x.train[,-1]
x.test  = x.test[,-1]
## end data wrangling

## fit our logistic regression
control = trainControl(method="none", number=10, classProbs=TRUE)
metric = "Accuracy"

fit.lr  = train(target~., data=x.train, method="glm", family="binomial", metric=metric, trControl=control)

## get our predictions/probabilities
probs.train = predict(fit.lr, x.train, type="prob")
probs.test = predict(fit.lr, x.test, type="prob")
open.ind = which(names(probs.train)=='open')

confusionMatrix(predict(fit.lr, x.test), x.test$target)


## get our fwd/backward sequences
seq.train = data.frame(probs2seq(probs.train[,open.ind]))
seq.train = cbind(x.train$target, seq.train)
names(seq.train)[1] = 'target'

seq.test = data.frame(probs2seq(probs.test[,open.ind]))
seq.test = cbind(x.test$target, seq.test)
names(seq.test)[1] = 'target'


fit.seq  = train(target~., data=seq.train, method="glm", family="binomial", metric=metric, trControl=control)

## analysis
confusionMatrix(predict(fit.seq, seq.test), seq.test$target)


## now for the combining
probs.test = probs.test[,open.ind]
probs.seq  = predict(fit.seq, seq.test, type="prob")[,open.ind]

lambda = 0.75
probs.combined = lambda * probs.test + (1-lambda)*probs.seq

preds.combined = factor(levels=c('closed', 'open'))
preds.combined[which(probs.combined >  0.5)] = 'open'
preds.combined[which(probs.combined <= 0.5)] = 'closed'

confusionMatrix(preds.combined, x.test$target)