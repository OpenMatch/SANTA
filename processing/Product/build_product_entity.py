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
import numpy as np

def split_data(dataset_path=None, locale="us", dev_size=0.05):
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

    df_examples_products = df_examples_products[df_examples_products[col_large_version] == 1]
    df_examples_products = df_examples_products[df_examples_products[col_product_locale] == locale]
    df_examples_products[col_gain] = df_examples_products[col_esci_label].apply(
        lambda esci_label: esci_label2gain[esci_label])

    list_query_id = df_examples_products[col_query_id].unique()  # 去除重复元素
    # 划分数据集
    list_query_id_train, list_query_id_dev = train_test_split(list_query_id, test_size=dev_size, random_state=10)

    df_train = df_examples_products[df_examples_products[col_query_id].isin(list_query_id_train)]
    df_dev = df_examples_products[df_examples_products[col_query_id].isin(list_query_id_dev)]

    return df_train, df_dev


def load_pretrain_data(df_examples_products):
    data_samples = []
    col_product_description = "product_description"
    col_gain = 'gain'
    col_product_bullet_point = 'product_bullet_point'
    col_product_title = "product_title"
    for (index, row) in tqdm(df_examples_products.iterrows()):
        description = str(row[col_product_description])
        bullet = str(row[col_product_bullet_point])
        if description != "None" and bullet != "None":
            querys = bullet.split("\n")
            query = random.sample(querys, 1)[0]
            soup = BeautifulSoup(description, 'html.parser')
            positive = soup.text
            if row[col_gain] == 1:
                if positive!=description and positive!="" and query!="":
                    #filter data without html
                    data_samples.append({'query': query, 'positive': positive,'title':row[col_product_title]})
    return data_samples

def creat_pretrain_data(data,outpath,if_train):
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    with open(outpath, 'w') as f:
        for idx, item in enumerate(tqdm(data)):
            replacelist = tokenizer.additional_special_tokens
            group = {}
            positives = []
            bullet=item["query"]
            description=item["positive"]
            title=item['title']
            if if_train:
                generate = description.split()
                if len(generate) < 2:
                    continue

                # 找到标题中的实体
                title_identity = []
                title_tokens = nltk.word_tokenize(title)
                title_tokens = nltk.pos_tag(title_tokens)
                for token in title_tokens:
                    if token[1] == 'NNP' or token[1] == 'NNPS':
                        title_identity.append(token[0])
                title_identity = [x.lower() for x in title_identity]

                indices = []
                description_identity = []
                des_tokens_raw = nltk.word_tokenize(description)
                des_tokens = nltk.pos_tag(des_tokens_raw)
                for idx, token in enumerate(des_tokens):
                    if token[1] == 'NNP' or token[1] == 'NNPS' or token[0].lower() in title_identity:
                        description_identity.append(token[0])
                        indices.append(idx)

                # 对标识符号列表去重并且保持顺序
                result = []
                my_dict = {}
                for idx, item in enumerate(description_identity):
                    if item not in my_dict:
                        my_dict[item] = 1
                        result.append(item)
                # 找出选定元素并建立映射
                result = result[:len(replacelist) - 1]
                replacelist_new = replacelist[:len(result)]
                mapping = dict(zip(result, replacelist_new))
                label = []

                # 把选定的元素替换为映射
                for idex, i in enumerate(description_identity):
                    if description_identity[idex] not in mapping:
                        continue
                    description_identity[idex] = mapping[description_identity[idex]]

                for i in range(len(result)):
                    label.append(replacelist[i])
                    label.append(result[i])
                label.append(replacelist[i + 1])

                # 把标识符映射到源代码段中,进行mask
                generate_mapping = dict(zip(indices, description_identity))
                for idx, item in enumerate(des_tokens_raw):
                    if idx in generate_mapping:
                        des_tokens_raw[idx] = generate_mapping[idx]

                labels = ' '.join(label)
                labels = tokenizer(labels, add_special_tokens=False, max_length=60, truncation=True, padding=False)[
                    'input_ids']
                group['labels'] = labels
                description = ' '.join(des_tokens_raw)

            positives.append(description)
            positives = tokenizer(positives, add_special_tokens=False, max_length=240, truncation=True, padding=False)[
                'input_ids']
            query = tokenizer.encode(bullet, add_special_tokens=False, max_length=50, truncation=True)
            if query==[]:
                continue
            if positives[0]==[]:
                continue
            group['query'] = query
            group['positives'] = positives
            group['negatives'] = []
            f.write(json.dumps(group) + '\n')
def main():
    parser = ArgumentParser()

    parser.add_argument('--input', type=str, default="./esci_data",
                        help="input data directory contains two files, shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet")
    parser.add_argument('--pretrain_train', type=str, default="./pretrain_train.jsonl")
    parser.add_argument('--pretrain_eval', type=str, default="./pretrain_eval_raw.jsonl",
                        help="evaluate on it to select best dev checkpoint during pretraining")


    args = parser.parse_args()
    random_seed = 42
    random.seed(random_seed)

    df_train, df_dev = split_data(args.input, "us", 0.05)
    train_data = load_pretrain_data(df_train)
    dev_data = load_pretrain_data(df_dev)
    with open(args.pretrain_eval, 'w') as file_object:
        json.dump(dev_data, file_object)
    creat_pretrain_data(train_data, args.pretrain_train, if_train=True)


if __name__ == '__main__':
    main()