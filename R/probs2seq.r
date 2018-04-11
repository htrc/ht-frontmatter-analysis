probs2seq = function(predictions) {
	fwd = rep(1, length(predictions));
	bwd = rep(1, length(predictions));
	
	fwd[1:(length(fwd)-1)] = predictions[2:length(predictions)]
	fwd[length(fwd)] = predictions[length(predictions)]
	
	bwd[2:length(fwd)] = predictions[1:(length(predictions)-1)]
	bwd[1] = predictions[1];
	
	return(cbind(bwd, fwd))
}