# Structure Aware Dense Retrieval (SANTA)

Source code and dataset for ACL 2023 Structure-Aware Language Model Pretraining Improves Dense Retrieval on Structured Data.

## Environment
(1) Install the following packages using Pip or Conda under this environment:
```
transformers==4.22.2
nltk==3.7
numpy==1.23.2
datasets>=2.4.0
tree-sitter==0.0.5
faiss==1.7.4
scikit-learn>=1.1.2
pandas==1.5.0
pytrec-eval==0.5
tensorboard
```
(2) install openmatch. To download OpenMatch as a library and obtain openmatch-thunlp-0.0.1.
```
git clone https://github.com/OpenMatch/OpenMatch.git
cd OpenMatch
pip install .
```

## Pretrained Checkpoint
#### HuggingFace Link
(1) The checkpoint of the pretrained SANTA model on `Python` data is here.
(2) The checkpoint of the pretrained SANTA model on `ESCI (large)` data is here.

#### Pretraining Paraments
```
learning_rate=5e-5
num_train_epochs=6
train_n_passages=1 
per_device_train_batch_size=16
```

## Data Download
#### Code
(1) [CodeSearchNet](https://github.com/github/CodeSearchNet)
```
git clone https://github.com/github/CodeSearchNet.git
cd CodeSearchNet/
script/setup
```

(2) [Adv](https://github.com/microsoft/CodeXGLUE/tree/main/Text-Code/NL-code-search-Adv)
```
wget https://github.com/microsoft/CodeXGLUE/raw/main/Text-Code/NL-code-search-Adv/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset AdvTest && cd AdvTest
wget https://zenodo.org/record/7857872/files/python.zip
unzip python.zip && python preprocess.py && rm -r python && rm -r *.pkl && rm python.zip
```

(3) [CodeSearch](https://github.com/microsoft/CodeBERT/tree/master/GraphCodeBERT/codesearch)
```
wget https://github.com/microsoft/CodeBERT/raw/master/GraphCodeBERT/codesearch/dataset.zip
unzip dataset.zip && rm -r dataset.zip && mv dataset CSN && cd CSN
bash run.sh 
```
#### Product
you can download ESCI data from here: [ESCI](https://github.com/amazon-science/esci-data/tree/main/shopping_queries_dataset)

## Process Data
#### Process Code 

(1) Collect pretraining code data.

The pretraining data for the code is sourced from the downloaded CodeSearchNet, which consists of six programming languages. The collection of pretraining data is based on the data statistics provided by [CodeT5](https://arxiv.org/abs/2109.00859) and [CodeRetriever](https://arxiv.org/abs/2201.10866). For the five programming languages other than Python, the `train` from CodeSearchNet will be merged to create the pretraining data `${PRETRAIN_RAW_PATH}/${pretrain_raw_data}`. However, for Python, both the `train` and `test` from CodeSearchNet will be merged as the pretraining data. When selecting a checkpoint for pretraining, the `valid` from CodeSearchNet will be used as the `dev` for all programming languages. 

(2) Process pretraining code data.

process the raw code pretraining data and make it suitable for pretraining inputs `<query, positive, label>`.
Enter the folder `shell` and run the shell script:
```
bash process-pretrain-code.sh
```

(3) Process finetuning code data.

For the `Adv` and `CodeSearch` two code retriever tasks, you can process the raw training file `train ` into input path `${FINETUNE_RAW_PATH}/${finetune_raw_data}`.
```
bash process-finetune-code.sh
```

#### Process Product

(1) Collect pretraining product data.

To use the ESCI (large) data for pretraining, please ensure that the following two files, `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet`, are downloaded and available in the pretraining path `${PRETRAIN_RAW_PATH}`.

(2) Process pretraining produc data.

processing the raw product pretraining data makes it suitable for pretraining inputs `<query, positive, label>` and save dev set into `${PRETRAIN_PATH}/${pretrain_eval_data}` for selecting pretraining checkpoint.
```
bash process-pretrain-product.sh
```

(3) Process finetuning produc data.

For the product search task, we use ESCI (small) data for finetuning. you can process the raw training file which includes  `shopping_queries_dataset_examples.parquet` and `shopping_queries_dataset_products.parquet` into input path `${FINETUNE_PATH}/${finetune_data}` for fintuning and eval path `${FINETUNE_PATH}/${finetune_eval_data}` for selecting finetuning checkpoint.

```
bash process-finetune-product.sh
```

## Pretraining
#### Pretraing for code search

To continue pretraining CodeT5 for different programming languages, utilize the corresponding processed code pretraining data. For instance, if you want to train a Python code retrieval model, only use Python pretraining data for training.
```
bash pretrain-code.sh
```

#### Pretraing for product search

Continuing pretraining T5 using the processed product pretraining data to get a product retrieval model.
```
bash pretrain-product.sh
```

## Finetuning
#### Finetuning for code search

For the pretrained checkpoint, finetuning is performed on downstream tasks related to code retrieval, such as  `Adv ` and `CodeSearch`, useing processed finetuned data. For example, If you have pretrained on Python code, you should also fine-tune on Python code.
```
bash finetune-code.sh
```

#### Finetuning for product search

For the product retrieval task, the pretrained checkpoint is fine-tuned on `ESCI (small)`, useing processed finetuned data. You can find more details about this task [here](https://github.com/amazon-science/esci-data).
```
bash finetune-product.sh
```
> P.S. If you want to use hard negatives, you need to set the parameter `train_n_passages` to n+1, where n is the number of hard negatives.

## Evaluating

Before evaluating the code and product retrieval tasks, it is necessary to download `OpenMatch`.
```
git clone https://github.com/OpenMatch/OpenMatch.git
```

#### Evaluating Code Retrieval

For code retrieval tasks, you need to generate test data to conform to OpenMatch's input.
```
bash build-code-test.sh
```

Then, you need to build the Faiss index and obtain the necessary files for inference.

```
bash index-code.sh
```

Evaluate using the obtained inference files.
```
bash evaluate_code.sh
```

#### Evaluating Product Retrieval

For product retrieval task, you need to generate test data to conform to OpenMatch's input.
```
bash build-product-test.sh
```

Encode the query and description of the product as embeddings and save them.
```
bash index-product.sh
```
Calculate scores for the encoded embeddings and sort them to obtain two files `hypothesis.results` and `test.qrels`, which will be used to calculate the NDCG score.
```
bash evaluate_product.sh
```