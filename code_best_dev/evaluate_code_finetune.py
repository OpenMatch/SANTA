import sys
import argparse
import logging
import os
import pickle
import random
import torch
import json
import numpy as np
from torch.nn import CrossEntropyLoss, MSELoss
from torch.utils.data import DataLoader, Dataset, SequentialSampler, RandomSampler, TensorDataset
from transformers import (WEIGHTS_NAME, AdamW, get_linear_schedule_with_warmup,
                          RobertaConfig, RobertaModel, RobertaTokenizer,T5ForConditionalGeneration)
from tqdm import tqdm
import glob
import shutil
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)



def encode_batch(batch_text, tokenizer, max_length):
    outputs = tokenizer.batch_encode_plus(
        batch_text,
        max_length=max_length,
        pad_to_max_length=True,
        return_tensors='pt',
        truncation=True,

    )
    input_ids = outputs["input_ids"]
    attention_mask = outputs["attention_mask"].bool()
    return input_ids, attention_mask



class Dataset(Dataset):
    def __init__(self, data,tokenizer,args):
          self.data=data
          self.tokenizer=tokenizer
          self.args=args
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
        js=self.data[i]
        if 'code_tokens' in js:
            code=' '.join(js['code_tokens'])
        else:
            code=' '.join(js['function_tokens'])
        nl=' '.join(js['docstring_tokens'])
        url=js['url']

        return {"code":code,
                "nl":nl,
                "url":url,
            }

class Code_Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        code = [ex["code"] for ex in batch]
        code_ids, code_masks = encode_batch(code, self.tokenizer, self.args.code_length)
        url = [ex["url"] for ex in batch]

        return {
            "code_ids": code_ids,
            "code_masks": code_masks,
            "url": url,
        }

class Query_Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        nl = [ex["nl"] for ex in batch]
        nl_ids, nl_masks = encode_batch(nl, self.tokenizer, self.args.nl_length)

        url = [ex["url"] for ex in batch]
        return {
            "nl_ids": nl_ids,
            "nl_masks": nl_masks,
            "url": url,
        }

def load_data(file_path=None):
    data=[]
    with open(file_path) as f:
        if "jsonl" in file_path:
            for line in f:
                line = line.strip()
                js = json.loads(line)
                if 'function_tokens' in js:
                    js['code_tokens'] = js['function_tokens']
                data.append(js)
        elif "codebase" in file_path or "code_idx_map" in file_path:
            js = json.load(f)
            for key in js:
                temp = {}
                temp['code_tokens'] = key.split()
                temp["retrieval_idx"] = js[key]
                temp['doc'] = ""
                temp['docstring_tokens'] = ""
                data.append(temp)
        elif "json" in file_path:
            for js in json.load(f):
                data.append(js)
    return data

def set_seed(seed=42):
    random.seed(seed)
    os.environ['PYHTONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

def evaluate(args, model, tokenizer, file_name,check_path,writer, best_dev_mrr,number):
    code_collator = Code_Collator(tokenizer, args)
    query_collactor=Query_Collator(tokenizer, args)

    codebase_file=load_data(args.codebase_file)
    file_name=load_data(file_name)

    code_dataset = Dataset(codebase_file,tokenizer, args)
    code_sampler = SequentialSampler(code_dataset)
    code_dataloader = DataLoader(code_dataset, sampler=code_sampler, batch_size=args.eval_batch_size, num_workers=4,collate_fn=code_collator)

    query_dataset = Dataset(file_name,tokenizer, args)
    query_sampler = SequentialSampler(query_dataset)
    query_dataloader = DataLoader(query_dataset, sampler=query_sampler, batch_size=args.eval_batch_size, num_workers=4,collate_fn=query_collactor)

    # Eval!
    logger.info("***** Running evaluation *****")
    logger.info("  Num queries = %d", len(query_dataset))
    logger.info("  Num codes = %d", len(code_dataset))
    logger.info("  Batch size = %d", args.eval_batch_size)

    model.eval()
    code_vecs = []
    nl_vecs = []
    for batch in tqdm(code_dataloader):
        code_inputs = batch["code_ids"].to(args.device)
        code_masks = batch["code_masks"].to(args.device)
        with torch.no_grad():
            decoder_input_ids = torch.zeros((code_inputs.shape[0], 1), dtype=torch.long).to(args.device)
            code_out = model(
                input_ids=code_inputs,
                attention_mask=code_masks,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            code_vec = code_out[2]
            code_vec = code_vec[-1].squeeze(1)
            code_vecs.append(code_vec.cpu().numpy())

    for batch in tqdm(query_dataloader):
        nl_inputs = batch["nl_ids"].to(args.device)
        nl_masks = batch["nl_masks"].to(args.device)
        with torch.no_grad():
            decoder_input_ids = torch.zeros((nl_inputs.shape[0], 1), dtype=torch.long).to(args.device)
            nl_out = model(
                input_ids=nl_inputs,
                attention_mask=nl_masks,
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
            )
            nl_vec = nl_out[2]
            nl_vec = nl_vec[-1].squeeze(1)
            nl_vecs.append(nl_vec.cpu().numpy())


    model.train()
    code_vecs = np.concatenate(code_vecs, 0)
    nl_vecs = np.concatenate(nl_vecs, 0)

    scores = np.matmul(nl_vecs, code_vecs.T)

    sort_ids = np.argsort(scores, axis=-1, kind='quicksort', order=None)[:, ::-1]

    nl_urls = []
    code_urls = []
    for i, example in tqdm(enumerate(query_dataset)):
        nl_urls.append(example["url"])

    for i, example in tqdm(enumerate(code_dataset)):
        code_urls.append(example["url"])

    ranks = []
    for url, sort_id in zip(nl_urls, sort_ids):
        rank = 0
        find = False
        for idx in sort_id[:1000]:
            if find is False:
                rank += 1
            if code_urls[idx] == url:
                find = True
        if find:
            ranks.append(1 / rank)
        else:
            ranks.append(0)

    result = {
        "eval_mrr": float(np.mean(ranks))
    }
    eval_mrr=float(np.mean(ranks))

    print("-------------------")
    print(3000 * number)
    print(best_dev_mrr)
    logger.info("***** Eval results *****")
    print(eval_mrr)
    if eval_mrr > best_dev_mrr:
        best_dev_mrr = eval_mrr
        best_path = args.model_name_or_path + "best_dev"
        shutil.copytree(check_path, best_path, dirs_exist_ok=True)
    writer.add_scalar('eval/mrr', eval_mrr, 3000 * number)

    return best_dev_mrr



def main():
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--eval_data_file", default=None, type=str,
                        help="An optional input evaluation data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--codebase_file", default="D:\\T5数据\\CSN\\codebase.jsonl", type=str,
                        help="An optional input test data file to codebase (a jsonl file).")
    parser.add_argument("--model_name_or_path", default="D:\\T5数据\\codet5-pretrian\\mask-all-pretrain\\best_dev", type=str,
                        help="The model checkpoint for weights initialization.")
    parser.add_argument("--nl_length", default=30, type=int,
                        help="Optional NL input sequence length after tokenization.")
    parser.add_argument("--code_length", default=100, type=int,
                        help="Optional Code input sequence length after tokenization.")
    parser.add_argument("--eval_batch_size", default=2, type=int,
                        help="Batch size for evaluation.")


    # print arguments
    args = parser.parse_args()
    # set log
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S', level=logging.INFO)
    # set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    args.device = device
    logger.info("device: %s, n_gpu: %s", device, args.n_gpu)
    writer = SummaryWriter(args.model_name_or_path)


    checkpoint_path = args.model_name_or_path
    len_path = len(checkpoint_path)
    print(len_path)
    id_file = checkpoint_path + '/checkpoint-*'
    print(id_file)
    len_file = len(glob.glob(id_file))
    print(f"all checkpoint amounts : {len_file}")
    best_dev_mrr = 0
    all_checkpoint = []
    for file in sorted(glob.glob(id_file), key=lambda name: int(name[len_path + 11:])):
        all_checkpoint.append(file)

    all_checkpoint.reverse()
    all_checkpoint = all_checkpoint[:]
    print(f"new checkpoint amounts : {all_checkpoint}")
    print(f"checkpoint list: {all_checkpoint}")
    for c in all_checkpoint:
        print(c)
    for i in range(len(all_checkpoint)):
        check_path = all_checkpoint[i]
        tokenizer = RobertaTokenizer.from_pretrained(check_path)
        model = T5ForConditionalGeneration.from_pretrained(check_path)
        logger.info("Training/evaluation parameters %s", args)
        model.to(args.device)
        # Evaluation
        results = {}
        model.to(args.device)
        result = evaluate(args, model, tokenizer, args.eval_data_file, check_path, writer, best_dev_mrr, i + 1)
        best_dev_mrr = result




if __name__ == "__main__":
    main()