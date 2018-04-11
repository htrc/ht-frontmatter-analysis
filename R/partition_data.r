
partition_data = function(x, k=10, folds_in_test_set=c(9,10)) {
volumes = unique(x$volume)

set.seed(7)
folds = createFolds(volumes, k=k)
x.train = x[FALSE,]
x.test  = x[FALSE,]

for(i in 1:k) {
	fold = folds[[i]]
	for(j in 1:length(fold)) {
		volume = volumes[fold[j]]
		if(! i %in% folds_in_test_set) {
			x.train = rbind(x.train, x[which(x$volume==volume),])
		} else {
			x.test = rbind(x.test, x[which(x$volume==volume),])
		}
	}
}


x.train = x.train[,-1]
x.test  = x.test[,-1]
split = list()
split[['train']] = x.train
split[['test']]  = x.test
return(split)

}
