# A DNABERT based Deep Learning Framework for Predicting Transcription Factor Binding Sites

1. **Introduction**
   
   We present a novel deep learning model called TFBS-Finder which is designed for predicting transcription factor binding sites (TFBSs) solely based on DNA sequences. The model comprises a pre-trained BERT module (DNABERT (kmer=5)), a convolutional neural network          (CNN) module, a modified convolutional block attention module (MCBAM), a multi-scale convolutions with attention (MSCA) module and an output module. The model is trained and tested on 165 ENCODE ChIP-seq datasets, demonstrating the superior ability of TFBS-Finder        to predict TFBSs. Experimental results indicate that TFBS-Finder achieves an average accuracy of 0.9301, a ROC-AUC of 0.961, and a PR-AUC of 0.961 for TFBS prediction. Here, the codes for implementing, training, and testing BERT-TFBS are provided.

2. **Python Environment**
 
   Python version           3.10.15
   
   torch                    2.5.1
   
   torchvision               0.20.1
  
   transformers              4.50.3
  
   numpy                     1.26.4
  
   pandas                    2.2.3
  
   scikit-learn              1.5.2
  
   matplotlib                3.9.3
  
   seaborn                   0.13.2
  
   tensorflow                2.15.1
  
   scipy                     1.14.1
  
   tqdm                      4.67.1

3. **LLM Environment**
   
   Conda environment   : trap

   Platform            : Linux 6.8.0-85-generic
  
   CUDA version        : 12.4
  
   GPU device          : NVIDIA GeForce GTX 1080 Ti

4. **How to run the model**

   The pre-trained BERT model is available at Huggingface as zhihan1996/DNA_bert_5

   To load the model from Huggingface, we can use the following code:
   
   import torch
   
   from transformers import AutoTokenizer, AutoModel
   
   tokenizer = AutoTokenizer.from_pretrained("zhihan1996/DNA_bert_5", trust_remote_code=True)
   
   model = AutoModel.from_pretrained("zhihan1996/DNA_bert_5", trust_remote_code=True)
   
   The training and testing will run together. The script is train_test.py which uses multiple GPU. This code will work for single GPU as well. 
   
   If you are running on 165 ChIP-seq datasets altogether for kmer = 5, please run train_test.py using:

   python train_test.py -k 5 -p 1 -q 29 -r 1

   where k is the size of kmer, p is the first TF folder, q is the 29th TF folder and r is the 1st subfolder dataset of each TF folder.
   
   The model is stored at https://drive.google.com/file/d/1h24GRS8_dxazusUgMP3u4DGrmIhbQr0f/view?usp=sharing

6. **script**
   
   a) **dataloader.py** converts DNA sequences into token embeddings.
   
   b) MCBAM.py implements the MCBAM module which integrates spatial attention and channel attention mechanisms.
   
   c) MSCA.py implements the MSCA module which uses multi-scale convolutions with an attention mechanism.
   
   d) model.py implements the TFBS-Finder which consists of a DNABERT module, a CNN module, an MCBAM, a MSCA module and an output module.
   
   e) MetricsHolder.py stores the performance metrics values.
   






   
