import json
from argparse import ArgumentParser
from transformers import T5Tokenizer, RobertaTokenizer, BertTokenizer
from tqdm import tqdm
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from bs4 import BeautifulSoup
import random
import nltk

def split_data(dataset_path=None, locale="us", dev_size=0.125):
    col_query = "query"  # the query of shop
    col_query_id = "query_id"
    col_product_id = "product_id"
    col_product_title = "product_title"
    col_product_locale = "product_locale"
    col_product_description = "product_description"
    col_product_bullet_point = 'product_bullet_point'
    col_esci_label = "esci_label"
    col_small_version = "small_version"
    col_large_version = "large_version"
    col_split = "split"
    col_gain = 'gain'
    df_examples = pd.read_parquet(os.path.join(dataset_path, 'shopping_queries_dataset_examples.parquet'))
    df_products = pd.read_parquet(os.path.join(dataset_path, 'shopping_queries_dataset_products.parquet'))
    # pd.read_parquet从文件路径加载一个parquet对象，返回一个DataFrame
    esci_label2gain = {
        'E': 1.0,
        'S': 0.1,
        'C': 0.01,
        'I': 0.0,
    }

    df_examples_products = pd.merge(
        df_examples,
        df_products,
        how='left',
        left_on=[col_product_locale, col_product_id],
        right_on=[col_product_locale, col_product_id]
    )

    df_examples_products = df_examples_products[df_examples_products[col_small_version] == 1]
    df_examples_products = df_examples_products[df_examples_products[col_split] == "train"]
    df_examples_products = df_examples_products[df_examples_products[col_product_locale] == locale]
    df_examples_products[col_gain] = df_examples_products[col_esci_label].apply(
        lambda esci_label: esci_label2gain[esci_label])

    list_query_id = df_examples_products[col_query_id].unique()
    list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size, random_state=10)
    df_examples_products = df_examples_products[
        [col_query_id, col_product_id, col_query, col_product_title, col_product_description, col_product_bullet_point,
         col_gain]]

    df_train = df_examples_products[df_examples_products[col_query_id].isin(list_query_id_train)]
    df_dev = df_examples_products[df_examples_products[col_query_id].isin(list_query_id_dev)]

    return df_train, df_dev


def load_finetune_data(data_path):
    data_samples = []
    col_query = "query"
    col_product_description = "product_description"
    col_product_title = "product_title"
    col_gain = 'gain'
    col_query_id = "query_id"
    col_product_id = "product_id"
    query2id = {}
    q_last = ''
    f = 'title :' + " {} " + 'text :' + " {}"

    for (index, row) in data_path.iterrows():
        if q_last != row[col_query]:
            query2id = {}
            query2id["query"] = row[col_query]
            query2id["query_id"] = row[col_query_id]
            query2id['products'] = []
            query2id['positive'] = []
            query2id['negative'] = []
            q_last = row[col_query]
            data_samples.append(query2id)
        i = str(row[col_product_description])
        if i != "None":
            soup = BeautifulSoup(i, 'html.parser')
            i = soup.text
            i = f.format(row[col_product_title], i)
            if row[col_gain] == 1:
                query2id['positive'].append({'text': i, 'gain': row[col_gain], "product_id": row[col_product_id]})
            else:
                query2id['negative'].append({'text': i, 'gain': row[col_gain], "product_id": row[col_product_id]})


        else:
            i = ''
            i = f.format(row[col_product_title], i)
            if row[col_gain] == 1:
                query2id['positive'].append({'text': i, 'gain': row[col_gain], "product_id": row[col_product_id]})
            else:
                query2id['negative'].append({'text': i, 'gain': row[col_gain], "product_id": row[col_product_id]})

    s = 0
    e = len(data_samples)
    while s < e:
        if len(data_samples[s]['negative']) == 0 or len(data_samples[s]['positive']) == 0:
            data_samples.pop(s)
        else:
            s += 1
        e = len(data_samples)


    return data_samples


def creat_finetune_data(data, output):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    with open(output, 'w') as f:
        for idx, item in enumerate(tqdm(data)):
            group = {}
            positives = []
            for p in item['positive']:
                positives.append(p['text'])
            negatives = []
            for n in item['negative']:
                negatives.append(n['text'])
            query = item['query']
            query = tokenizer.encode(query, add_special_tokens=False, max_length=20, truncation=True)
            positives = tokenizer(positives, add_special_tokens=False, max_length=256, truncation=True, padding=False)[
                'input_ids']
            negatives = tokenizer(negatives, add_special_tokens=False, max_length=256, truncation=True, padding=False)[
                'input_ids']

            group['query'] = query
            group['positives'] = positives
            group['negatives'] = negatives
            f.write(json.dumps(group) + '\n')
def main():
    parser = ArgumentParser()

    parser.add_argument('--input', type=str, default="./esci_data",
                        help="input data directory contains two files, shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet")

    parser.add_argument('--finetune_train', type=str, default="./finetune_train.jsonl")
    parser.add_argument('--finetune_eval', type=str, default="./finetune_eval_raw.jsonl",
                        help="evaluate on it to select best dev checkpoint during finetuning")


    args = parser.parse_args()
    df_train, df_dev = split_data(args.input, "us", 0.125)
    train_data = load_finetune_data(df_train)
    dev_data = load_finetune_data(df_dev)
    with open(args.finetune_eval, 'w') as file_object:
        json.dump(dev_data, file_object)

    creat_finetune_data(train_data, args.finetune_train)




if __name__ == '__main__':
    main()