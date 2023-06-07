import json
from argparse import ArgumentParser
from transformers import T5Tokenizer
from tqdm import tqdm
from transformers import RobertaTokenizer,BertTokenizer

def load_data(file_path=None):
    data=[]
    h=0
    with open(file_path) as f:
        for line in f:
            h+=1
            line=line.strip()
            js=json.loads(line)
            data.append(js)
        print(h)
    return data
def main():

    parser = ArgumentParser()
    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    data=load_data(args.input)
    with open(args.output, 'w') as f:
        defs1=0
        defs2=0
        for idx, item in enumerate(tqdm(data)):
            if 'code_tokens' in item:
                code = ' '.join(item['code_tokens'])
                defs1 += 1
            else:

                code = ' '.join(item['function_tokens'])
                defs2 += 1
            nl=' '.join(item['docstring_tokens'])
            group = {}
            positives = []

            positives.append(code)
            positives = tokenizer(positives, add_special_tokens=False, max_length=256, truncation=True, padding=False)['input_ids']
            query = tokenizer.encode(nl, add_special_tokens=False, max_length=50, truncation=True)
            n=tokenizer.convert_ids_to_tokens(query)
            p=tokenizer.convert_ids_to_tokens(positives[0])
            group['query'] = query
            group['positives'] = positives
            group['negatives'] = []

            f.write(json.dumps(group) + '\n')
    print(defs1)
    print(defs2)
if __name__ == '__main__':
    main()