# TFBS-Finder
Repository of the paper "TFBS-Finder: A DNABERT-based Deep Learning Model for Predicting Transcription Factor Binding Sites"

train_test.py uses multiple GPU. This code can be modified for single GPU as well.

If you are running on 165 ChIP-seq datasets altogether for kmer = 5, please use train_test.py using python train_test.py -k 5 -p 1 -q 29 -r 1 where k is the size of kmer, p is the first TF folder, q is the 29th TF folder and r is the 1st subfolder dataset of each TF folder
