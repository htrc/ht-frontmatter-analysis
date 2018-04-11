library(caret)
library(e1071)


x = read.csv("./data/java_training_data.csv", sep=" ", header=T)
x = x[,-1]

# a priori probability of guessing the correct label
prior = length(which(x$target=='open'))/length(x$target)

# joint probability distribution between x (this) and b (prev)
# 1 -> open, 0 -> closed
openInds = which(x$target=='open')
deltas = openInds[2:length(openInds)] - openInds[1:length(openInds)-1]
#deltas = openInds[1:length(openInds)-1] - openInds[2:length(openInds)]

px1b1 = (length(which(deltas==1))) / length(deltas)

x1b1 = 0
x0b1 = 0
x1b0 = 0
x0b0 = 0

for(i in 2:nrow(x)) {
	prev = x$target[i-1]
	curr = x$target[i]
	
	if(curr == 'open') {
		if(prev == 'open') {
			x1b1 = x1b1 + 1
		} else {
			x1b0 = x1b0 + 1
		}
	} else {
		if(prev == 'open') {
			x0b1 = x0b1 + 1
		} else {
			x0b0 = x0b0 + 1
		}	
	}
}

dist = matrix(nrow=2, ncol=2)
dist[1,1] = x1b1; dist[1,2] = x1b0; dist[2,1] = x0b1; dist[2,2] = x0b0
