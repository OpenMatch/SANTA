import torch
import numpy as np
import argparse
from torch.utils.data import DataLoader,SequentialSampler
from torch.utils.tensorboard import SummaryWriter
import glob
import json
from tqdm import tqdm
from torch.utils.data import Dataset
from transformers import T5Tokenizer, T5ForConditionalGeneration
import logging
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
    return input_ids, attention_mask


def load_data(file_path=None):
    data = []
    with open(file_path) as f:
        for line in f:
            line = line.strip()
            js = json.loads(line)
            data.append(js)
    data=data[0]
    return data

class Dataset(Dataset):
    def __init__(self,data,args):
          self.data=data
          self.args=args
    def __len__(self):
        return len(self.data)
    def __getitem__(self, i):
         data=self.data[i]
         query=data["query"]
         positive=data["positive"]
         return{"query":query,
             "positive":positive,
             }

class Collator(object):
    def __init__(self, tokenizer, args):
        self.tokenizer = tokenizer
        self.args = args

    def __call__(self, batch):
        query = [ex['query'] for ex in batch]
        query_ids, query_masks = encode_batch(query, self.tokenizer, self.args.query_block_size)
        positive = [ex['positive'] for ex in batch]
        positive_ids, positive_masks = encode_batch(positive, self.tokenizer, self.args.block_size)


        return {
            "query_ids": query_ids,
            "positive_ids": positive_ids,
            "query_masks": query_masks,
            "positive_masks": positive_masks,

        }

def test(opt,model,tokenizer,dataset,collator,best_dev_mrr,checkpoint_path,check_path,writer,number,eval_step):
    sampler = SequentialSampler(dataset)
    eval_dataloader = DataLoader(dataset,
                                 sampler=sampler,
                                 batch_size=opt.per_gpu_batch_size,
                                 drop_last=False,
                                 num_workers=0,
                                 collate_fn=collator
                                 )
    query_vecs = []
    positive_vecs = []
    model.eval()
    model = model.module if hasattr(model, "module") else model
    with torch.no_grad():
        print("start eval")
        for i, batch in tqdm(enumerate(eval_dataloader)):
            query_inputs = batch["query_ids"]
            positive_inputs = batch["positive_ids"]
            query_masks = batch["query_masks"]
            positive_masks = batch["positive_masks"]

            decoder_input_ids = torch.zeros((query_inputs.shape[0], 1), dtype=torch.long)
            query_out = model(
                input_ids=query_inputs.cuda(),
                attention_mask=query_masks.cuda(),
                decoder_input_ids=decoder_input_ids.cuda(),
                output_hidden_states=True,
            )
            query_vec = query_out[2]
            query_vec = query_vec[-1].squeeze(1)

            decoder_input_ids = torch.zeros((positive_inputs.shape[0], 1), dtype=torch.long)
            positive_out = model(
                input_ids=positive_inputs.cuda(),
                attention_mask=positive_masks.cuda(),
                decoder_input_ids=decoder_input_ids.cuda(),
                output_hidden_states=True,
            )
            positive_vec = positive_out[2]
            positive_vec = positive_vec[-1].squeeze(1)

            query_vecs.append(query_vec.cpu().numpy())
            positive_vecs.append(positive_vec.cpu().numpy())

        query_vecs = np.concatenate(query_vecs, 0)
        positive_vecs = np.concatenate(positive_vecs, 0)  # (9604,768) dev 集合里有9024个nl-code 对
        scores = np.matmul(query_vecs, positive_vecs.T)  # (9604,9024)(每个nl和 code 点乘后的结果)
        ranks = []
        for i in range(len(scores)):
            score = scores[i, i]  # nl和其对应的code分数,对角线
            rank = 1
            for j in range(len(scores)):
                if i != j and scores[i, j] >= score:
                    rank += 1
            ranks.append(1 / rank)


        eval_mrr=np.mean(ranks)
        print("-------------------")
        print(best_dev_mrr)
        print(eval_mrr)
        if eval_mrr>best_dev_mrr:
            best_dev_mrr=eval_mrr
            best_path=checkpoint_path+"best_dev"
            shutil.copytree(check_path, best_path,dirs_exist_ok=True)
        writer.add_scalar('eval/mrr', eval_mrr, 4000*eval_step)

        return best_dev_mrr
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    ## Required parameters
    parser.add_argument("--tensorboard_path", default=None, type=str,)
    parser.add_argument("--model_path", default=None, type=str, help="The directory of checkpoints.")
    parser.add_argument("--test_data", default=None, type=str,help="Product dev data file to evaluate the MRR(a jsonl file).")
    parser.add_argument("--per_gpu_batch_size", default=64, type=int)
    parser.add_argument("--block_size", default=240, type=int,help="Length of the product descriptions.")
    parser.add_argument("--query_block_size", default=50, type=int, help="Length of the product bullet points.")
    parser.add_argument("--eval_step", default=4000, type=int, help="How much steps to save the checkpoints")

    # print arguments
    opt = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    opt.n_gpu = torch.cuda.device_count()
    opt.device = device

    writer = SummaryWriter(opt.tensorboard_path)
    logger.info("Start eval")
    step, best_dev_mrr = 0, 0.0
    checkpoint_path = opt.model_path
    len_path=len(checkpoint_path)
    print(len_path)
    id_file = checkpoint_path + '/checkpoint-*'
    print(id_file)
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
        tokenizer = T5Tokenizer.from_pretrained(check_path)
        model = T5ForConditionalGeneration.from_pretrained(check_path)
        collator = Collator(tokenizer, opt)
        model = model.to(opt.device)

        logger.info(f"now dev checkpoints: {check_path}")
        print(check_path)
        best_dev_mrr=test(opt,
             model,
             tokenizer,
             test_dataset,
             collator,
             best_dev_mrr,
             checkpoint_path,
             check_path,
             writer,
             i+1,
             opt.eval_step
             )