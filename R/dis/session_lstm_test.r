library(caret)
library(e1071)
library(rnn)
library(keras)
library(tensorflow)
library(sigmoid)

#install_keras()

data(iris)
iris$Species = as.numeric(iris$Species)
iris = as.matrix(iris)
varnames = dimnames(iris)[[2]]
dimnames(iris) = NULL

x = iris[,-5]
y = iris[, 5]

# test/train split
ind = sample(2, nrow(iris), replace=T, prob=c(2/3, 1/3))
x.train  = x[which(ind==1),]
y.train  = y[which(ind==1)]
x.test   = x[which(ind==2),]
y.test   = y[which(ind==2)]
