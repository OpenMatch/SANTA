
import torch
import numpy as np
from torch.utils.data import DataLoader, RandomSampler, DistributedSampler, SequentialSampler,Dataset
import argparse
import pytrec_eval
import glob
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import json
from transformers import T5Tokenizer, T5ForConditionalGeneration,T5Model
from transformers import RobertaTokenizer, RobertaModel,BertTokenizer, BertModel
import logging
import os
import shutil
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
    return input_ids, attention_mask, outputs

class Dataset(Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        data = self.data[i]
        query = data["query"]
        products = data["positive"]+data["negative"]
        query_id = data["query_id"]

        return {"query": query,
                "products": products,
                "query_id": query_id,
                }


class Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):

        eval_query = [ex["query"] for ex in batch]
        eval_q_ids, eval_q_masks, out_q = encode_batch(eval_query, self.tokenizer, self.args.shop_query)

        eval_ = [t['text'] for example in batch for t in example["products"]]
        eval_ids, eval_masks, out = encode_batch(eval_, self.tokenizer, self.args.shop_product)

        query_id = [ex["query_id"] for ex in batch]
        query_id = query_id[0]

        product_id = [t['product_id'] for example in batch for t in example["products"]]
        label = []
        for example in batch:
            for t in example["products"]:
                if int(t['gain']) == 1:
                    label.append(1)
                else:
                    label.append(0)

        return {
            "eval_q_ids": eval_q_ids,
            "eval_q_masks": eval_q_masks,
            "eval_ids": eval_ids,
            "eval_masks": eval_masks,
            "label": label,
            "query_id": query_id,
            "product_id": product_id,
            "out": out,

        }

def load_data(file_path=None):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)
    data=data[0]
    return data

def test(opt,model,tokenizer,dataset,collator,best_dev_ndcg,checkpoint_path,check_path,writer,number):
    sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset,
                                 sampler=sampler,
                                 batch_size=1,
                                 drop_last=False,
                                 num_workers=0,
                                 collate_fn=collator
                                 )
    model.eval()
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        print("start eval")
        qrel = {}
        run = {}
        for i, batch in tqdm(enumerate(eval_dataloader)):
            label = batch["label"]
            query_inputs = batch["eval_q_ids"]
            inputs = batch["eval_ids"]
            query_masks = batch["eval_q_masks"]
            masks = batch["eval_masks"]
            query_id = batch["query_id"],
            product_id = batch["product_id"],

            query_id = query_id[0]
            product_id = product_id[0]

            decoder_input_ids = torch.zeros((query_inputs.shape[0], 1), dtype=torch.long)
            query_out = model(
                # code_out=logits,past_key_values,decoder_hidden_states,encoder_last_hidden_state,encoder_hidden_states
                input_ids=query_inputs.cuda(),
                attention_mask=query_masks.cuda(),
                decoder_input_ids=decoder_input_ids.cuda(),
                output_hidden_states=True,
            )
            query_vec = query_out[2]  # 1+12，初始嵌入输出外加每层输出处的隐状态
            query_vec = query_vec[-1].squeeze(1)  # -1，最后一层，shape (batch_size,1,768)-> #(batch_size,768)

            decoder_input_ids = torch.zeros((inputs.shape[0], 1), dtype=torch.long)
            product_out = model(
                input_ids=inputs.cuda(),
                attention_mask=masks.cuda(),
                decoder_input_ids=decoder_input_ids.cuda(),
                output_hidden_states=True,
            )

            product_vec = product_out[2]
            product_vec = product_vec[-1].squeeze(1)
            score = (product_vec * query_vec).sum(-1)
            score = score.tolist()  # score
            # 构建评测序列
            doc = {}
            la = {}
            for i in range(len(score)):
                doc[product_id[i]] = score[i]
                la[product_id[i]] = label[i]

            run[str(query_id)] = doc
            qrel[str(query_id)] = la

        evaluator = pytrec_eval.RelevanceEvaluator(qrel, {'ndcg_cut'})
        result = evaluator.evaluate(run)
        query_n = 0
        ndcg_100 = 0
        for k in result.keys():
            query_n += 1
            ndcg_100 += result[k]["ndcg_cut_100"]
        ndcg_100 = ndcg_100 / query_n
        print("-------------------")
        print(best_dev_ndcg)
        print(ndcg_100)
        if ndcg_100 > best_dev_ndcg:
            best_dev_ndcg = ndcg_100
            best_path = checkpoint_path + "best_dev"
            shutil.copytree(check_path, best_path, dirs_exist_ok=True)
        writer.add_scalar('eval/ndcg', ndcg_100, opt.eval_step * number)
        return best_dev_ndcg
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tensorboard_path", default=None, type=str,)
    parser.add_argument("--model_path", default=None, type=str, help="The directory of checkpoints.")
    parser.add_argument("--test_data", default=None, type=str,help="Product dev data file to evaluate the NDCG(a jsonl file).")
    parser.add_argument("--per_gpu_batch_size", default=1, type=int,help="The batch size must be 1")
    parser.add_argument("--shop_product", default=256, type=int,help="Length of the product descriptions.")
    parser.add_argument("--shop_query", default=20, type=int, help="Length of the product bullet points.")
    parser.add_argument("--eval_step", default=4000, type=int, help="How much steps to save the checkpoints")

    # print arguments
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.n_gpu = torch.cuda.device_count()
    opt.device = device
    writer = SummaryWriter(opt.tensorboard_path)

    logger.info("Start eval")
    step, best_dev_ndcg = 0, 0.0
    checkpoint_path = opt.model_path
    len_path=len(checkpoint_path)
    print(len_path)
    id_file = checkpoint_path + '/checkpoint-*'
    len_file = len(glob.glob(id_file))
    print(f"all checkpoint amounts : {len_file}")

    all_checkpoint = []
    for file in sorted(glob.glob(id_file), key=lambda name: int(name[len_path+11:])):
        all_checkpoint.append(file)

    all_checkpoint.reverse()
    all_checkpoint = all_checkpoint[:]
    print(f"new checkpoint amounts : {all_checkpoint}")
    print(f"checkpoint list: {all_checkpoint}")
    for c in all_checkpoint:
        print(c)

    test_data = load_data(opt.test_data)
    test_dataset = Dataset(test_data, opt)


    for i in range(len(all_checkpoint)):
        check_path = all_checkpoint[i]
        print(check_path)
        tokenizer = T5Tokenizer.from_pretrained(check_path)
        model = T5ForConditionalGeneration.from_pretrained(check_path)
        collator = Collator(tokenizer, opt)

        model = model.to(opt.device)
        logger.info(f"now dev checkpoints: {check_path}")
        best_dev_ndcg=test(opt,
             model,
             tokenizer,
             test_dataset,
             collator,
             best_dev_ndcg,
             checkpoint_path,
             check_path,
             writer,
             i+1
             )
