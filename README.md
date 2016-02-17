# RST Discourse Paring using Deep Neural Net
Implementations of RST discourse paring models represented in "Recursive Deep Models for Discourse Parsing" and "When Are Tree Structures Necessary for Deep Learning of Representations? ". Bi-directional LSTMs are applied to EDU sequences and Tree LSTMs are applied for tree construction.

## Requirements:
GPU 

matlab >= 2014b

For any pertinent question, feel free to contact jiweil@stanford.edu

## Training
run binary/discourse_binary.m
run multi/discourse_multi.m
## Testing
run /test/decode_beam_standard.m.


download [data,embeddings](http://cs.stanford.edu/~bdlijiwei/discourse_data.tar)

