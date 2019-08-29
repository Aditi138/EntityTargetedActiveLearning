# Active Learning for Entity Recognition

### Requirements
python 2.7 <br>
DynetVersion commit 284838815ece9297a7100cc43035e1ea1b133a5


### Data
In the ```data/```, create a directory per language as shown for ```data/Spanish```. Download the CoNLL train/dev/test NER datasets for that language here. To acquire LDC datasets, please get the required access.


For storing the trained models, create directory ```saved_models``` in the parent folder.
### Embeddings
Combine monolingual data acquired from Wikipedia with the plain text extracted from the labeled data. Train 100-d [Glove]((https://nlp.stanford.edu/projects/glove/)) embeddings

### Active Learning Simulation 
The best NER performance was obtained using fine-tuning training scheme. The scripts below runs simulation active learning runs for different active learning strategies:
``` cd commands``` <br>
*  ETAL + Partial-CRF + CT (Proposed recipe) <br> ``` ./ETAL_PARTIAL_CRF_CT.sh ```<br>
* ETAL + Full-CRF + CT <br>``` ./ETAL_FULL_CRF_CT.sh ```<br>
* CFEAL + Full-CRF + CT <br>``` ./CFEAL_PARTIAL_CRF_CT.sh ```<br>
* SAL + CT <br>
``` ./SAL_CT.sh ```<br>
Things to note:

We load the vocabulary from the following path```--aug_lang_train_path```. Therefore, create a conll formatted file with dummy labels from the unlabeled text.
For our experiments, we concatenated the transferred data with the unlabeled data (which was the entire training dataset) into a single conll formatted file. 
The conll format is a tab separated two-column format as shown below: <br>

```El   O``` <br>
```grupo    O```<br>

The LDC NER label set differ from the CoNLL label set by one tag. Therefore, add ``` --misc ``` to the argument set when running any experiments on CoNLL data. The label set has been hard-coded in the ```data_loaders/data_loader.py``` file. 

### Cross-Lingual Transferred Data
We used the model proposed by (Xie et al. 2018) to get the cross-lingually transferred data from English. 
Please refer to their code [here](https://github.com/thespectrewithin/cross-lingual_NER).

For the Fine-Tune training scheme, train a base NER model on the transferred model as follows:

    MODEL_NAME="spanish_full_transfer_baseline"
    python -u ../main.py \
        --dynet-seed 3278657 \
        --word_emb_dim 100 \
        --batch_size 10 \
        --model_name ${MODEL_NAME} \
        --lang es \
        --fixedVocab \
        --test_conll \
        --tot_epochs 1000 \
	--aug_lang_train_path $DATA/vocab.conll \
        --init_lr 0.015 \
        --valid_freq 1300 \
        --misc \
        --pretrain_emb_path $DATA/esp.vec \
        --dev_path $DATA/esp.dev \
        --test_path $DATA/esp.test \
        --train_path $DIR/transferred_data.conll  2>&1 | tee ${MODEL_NAME}.log 
        
### References
If you make use of this software for research purposes, we will appreciate citing the following:
```
@inproceedings{chaudhary19emnlp,
    title = {A Little Annotation does a Lot of Good: A Study in Bootstrapping Low-resource Named Entity Recognizers},
    author = {Aditi Chaudhary and Jiateng Xie and Zaid Sheikh and Graham Neubig and Jaime Carbonell},
    booktitle = {Conference on Empirical Methods in Natural Language Processing (EMNLP)},
    address = {Hong Kong},
    month = {November},
    url = {http://arxiv.org/abs/1908.08983},
    year = {2019}
}
```

### Contact
For any issues, please feel free to reach out to `aschaudh@andrew.cmu.edu`.
