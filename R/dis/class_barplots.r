x = read.table("data/all_pages_binaryclass.txt")[,2:3]
names(x) = c("seq", "class")

fps = vector()
fns = vector()

n.max = 30
for(n in 1:n.max) {
	x.exposed = x[which(x$seq <= n),]
	x.closed  = x[which(x$seq >  n),]
	
	tp = length(which(x.exposed$class == "open"))
	fp = nrow(x.exposed) - tp
	
	tn = length(which(x.closed$class  == "closed"))
	fn = nrow(x.closed) - tn
	
	fps[n] = fp
	fns[n] = fn
}

par(mar = c(5,5,2,5))
plot(fps, xlab="Number of Pages Exposed", ylab="False Positives", col="blue")
arrows(4, 3000, 5, fps[5], angle=0)
text(4, 3200, labels=c("171 false pos"), cex=0.65, col="blue")
par(new=T)
plot(fns, axes=F, xlab="", ylab="", col="red")
axis(side=4)
mtext(side=4, line=3, 'False Negatives')
legend(10, 18000, legend=c("False Positives", "False Negatives"), pch=c(1,1), col=c("blue", "red"), cex=0.75)

