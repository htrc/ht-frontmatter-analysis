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
probs.features.train = predict(fit.lr, x.train, type="prob")
probs.features.test = predict(fit.lr, x.test, type="prob")
open.ind = which(names(probs.features.train)=='open')

prob.open.features.train = probs.features.train[,open.ind]
prob.closed.features.train = 1 - prob.open.features.train
prob.open.features.test  = probs.features.test[,open.ind]
prob.closed.features.test = 1 - prob.open.features.test

confusionMatrix(predict(fit.lr, x.test), x.test$target)


## get our fwd/backward sequences
context.train = data.frame(probs2seq(prob.open.features.train))
context.train = cbind(x.train$target, context.train)
names(context.train)[1] = 'target'

context.test = data.frame(probs2seq(prob.open.features.test))
context.test = cbind(x.test$target, context.test)
names(context.test)[1] = 'target'


fit.seq  = train(target~., data=context.train, method="glm", family="binomial", metric=metric, trControl=control)

probs.context.train = predict(fit.seq, context.train, type="prob")
probs.context.test = predict(fit.seq, context.test, type="prob")

prob.open.context.train = probs.context.train[,open.ind]
prob.closed.context.train = 1 - prob.open.context.train
prob.open.context.test  = probs.context.test[,open.ind]
prob.closed.context.test = 1 - prob.open.context.test




## now for the combining
alpha = seq(0, 1, by=0.05)

for(i in 1:length(alpha)) {
prob.open.combined.train = (prob.open.features.train)^alpha[i] *
	(prob.open.context.train)^(1-alpha[i])
prob.closed.combined.train = (prob.closed.features.train)^alpha[i] *
	(prob.closed.context.train)^(1-alpha[i])
final.train = prob.open.combined.train / 1
#	(prob.open.combined.train + prob.closed.combined.train)

# linear combination
#prob.open.combined.train = (prob.open.features.train)*alpha[i] +
#	(prob.open.context.train)*(1-alpha[i])
#final.train = prob.open.combined.train
# end linear combination

predictions = factor(levels=c('closed', 'open'))
predictions[which(final.train >  0.5)] = 'open'
predictions[which(final.train <= 0.5)] = 'closed'

cm = confusionMatrix(predictions, x.train$target)

accuracy = sum(diag(cm$table)) / sum(cm$table)
print(paste(c("alpha: ", alpha[i], "acc:   ", accuracy)))
}

max_alpha = 0.75

prob.open.combined.test = (prob.open.features.test)^max_alpha *
	(prob.open.context.test)^(1-max_alpha)
prob.closed.combined.test = (prob.closed.features.test)^max_alpha *
	(prob.closed.context.test)^(1-max_alpha)
final.test = prob.open.combined.test / 1
#	(prob.open.combined.test + prob.closed.combined.test)
	

# linear combination
#prob.open.combined.test = (prob.open.features.test)*alpha[i] +
#	(prob.open.context.test)*(1-alpha[i])
#final.test = prob.open.combined.test
# end linear combination

predictions = factor(levels=c('closed', 'open'))
predictions[which(final.test >  0.5)] = 'open'
predictions[which(final.test <= 0.5)] = 'closed'

confusionMatrix(predictions, x.test$target)