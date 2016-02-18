# RST Discourse Paring using Deep Neural Net
Implementations of RST discourse paring models represented in "Recursive Deep Models for Discourse Parsing" and "When Are Tree Structures Necessary for Deep Learning of Representations? ". Bi-directional LSTMs are applied to EDU sequences and Tree LSTMs are applied for tree construction.

## Requirements:
GPU 

matlab >= 2014b

For any pertinent question, feel free to contact jiweil@stanford.edu

##Folders
Binary:  a binary structure
classifier to determine whether two adjacent text
units should be merged to form a new subtree.

Multi: a multi-class classifier to determine which discourse
relation label should be assigned to the new subtree.

Infer: Doing inference on testing dataset.

## Training
run binary/discourse_binary.m

run multi/discourse_multi.m
## Testing
infer/Evaluation.m

download [data,embeddings](http://cs.stanford.edu/~bdlijiwei/discourse_data.tar)

```latex

@inproceedings{li2014recursive,
    title={Recursive Deep Models for Discourse Parsing.},
    author={Li, Jiwei and Li, Rumeng and Hovy, Eduard H},
    booktitle={EMNLP},
    pages={2061--2069},
    year={2014}
}

@article{li2015tree,
    title={When are tree structures necessary for deep learning of representations?},
    author={Li, Jiwei and Jurafsky, Dan and Hovy, Eudard},
    journal={arXiv preprint arXiv:1503.00185},
    year={2015}
}

```
