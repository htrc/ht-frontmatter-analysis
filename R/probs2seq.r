# takes a matrix of probabilities of class membership (each row
# corresponds to an instance, and the ith column is the probability
# predicted for class i...Returns a new matrix where the jth
# row consists of the predictions for the PREVIOUS (j-1th) and
# NEXT (j+1th) instances so that these can subsequently be used
# as features for instance j.
# N.B. dumb hacks at the first and last instances (rows).
 
probs2seq = function(predictions) {
	fwd = rep(1, length(predictions));
	bwd = rep(1, length(predictions));
	
	fwd[1:(length(fwd)-1)] = predictions[2:length(predictions)]
	fwd[length(fwd)] = predictions[length(predictions)]
	
	bwd[2:length(fwd)] = predictions[1:(length(predictions)-1)]
	bwd[1] = predictions[1];
	
	return(cbind(bwd, fwd))
}
