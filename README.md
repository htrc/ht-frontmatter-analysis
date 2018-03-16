# Initial Findings Regarding Front Matter Identification in HT

My goal in this exploratory work was to assess the overall feasibility
of applying machine learning techniques to the problem of discriminating
between *factual* and *creative* content in the initial pages of
Hathi Trust (HT) volumes.  The basic problem statement runs like so:
Given a HT volume *V*, analyze the first *n* pages of *V* (where *n*
is a small integer, on the order of 5-30).  For each of these *n* pages
*p<sub>1</sub>, p<sub>2</sub>, ..., p<sub>n</sub>*, we seek a prediction
*p<sup>'</sup><sub>i</sub>* in *{factual, creative}* that can guide
a decision whether or not to expose *p<sub>i</sub>* to the public.

More specifically, I pursued several 
questions:
  * 
