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
To process the raw code pretraining data and make it suitable for pretraining inputs `<query, positive, label>`.
Enter the folder `shell` and run the shell script:
```
bash process-pretrain-code.sh
```
(3) Process finetuning code data.
For the Adv and CodeSearch tasks, you can process the raw training data  `train ` into input data.
