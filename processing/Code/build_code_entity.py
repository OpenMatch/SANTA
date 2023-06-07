from transformers import T5Tokenizer, T5ForConditionalGeneration,RobertaTokenizer
import json
import tokenize
import keyword
from io import BytesIO
from argparse import ArgumentParser
from tqdm import tqdm
import random
import tree_sitter
import os
import sys
php_keylist=[ #php key words.
  "abstract",
  "and",
  "array",
  "as",
  "break",
  "callable",
  "case",
  "catch",
  "class",
  "clone",
  "const",
  "continue",
  "declare",
  "default",
  "die",
  "do",
  "echo",
  "else",
  "elseif",
  "empty",
  "enddeclare",
  "endfor",
  "endforeach",
  "endif",
  "endswitch",
  "endwhile",
  "eval",
  "exit",
  "extends",
  "false"
  "final",
  "finally",
  "for",
  "foreach",
  "function",
  "global",
  "goto",
  "if",
  "implements",
  "include",
  "include_once",
  "instanceof",
  "insteadof",
  "interface",
  "isset",
  "list",
  "namespace",
  "new",
  "null",
  "or",
  "parent",
  "print",
  "private",
  "protected",
  "public",
  "require",
  "require_once",
  "return",
  "static",
  "self",
  "switch",
  "this",
  "true",
  "throw",
  "trait",
  "try",
  "unset",
  "use",
  "var",
  "while",
  "xor",
  "__CLASS__",
  "__DIR__",
  "__FILE__",
  "__FUNCTION__",
  "__LINE__",
  "__METHOD__",
  "__NAMESPACE__",
  "__TRAIT__",
]
def delete_elements(arr):
    delete_count = int(0.5 * len(arr))
    deleted = [False] * len(arr)
    while delete_count > 0:
        idx = random.randint(0, len(arr) - 1)
        if deleted[idx]:
            continue
        deleted[idx] = True
        for i in range(len(arr)):
            if arr[i] == arr[idx]:
                deleted[i] = True
                delete_count -= 1
    new_arr = []
    for i in range(len(arr)):
        if not deleted[i]:
            new_arr.append(arr[i])
    deleted_elements=[]
    for i in range(len(arr)):
        if deleted[i]:
            deleted_elements.append(arr[i])

    return deleted_elements,new_arr

def flatten(arr):
    res = []
    for i in arr:
        if isinstance(i, list):
            res.extend(flatten(i))
        else:
            res.append(i)
    return res

def index_to_code_token(index, code):

    start_point = index[0]
    end_point = index[1]
    if start_point[0] == end_point[0]:
        s = code[start_point[0]][start_point[1]:end_point[1]]
    else:
        s = ""
        s += code[start_point[0]][start_point[1]:]
        for i in range(start_point[0]+1, end_point[0]):
            s += code[i]
        s += code[end_point[0]][:end_point[1]]
    return s

def tree_to_token_index(root_node):

    if (len(root_node.children) == 0 or root_node.type.find('string') != -1):
        return [(root_node.start_point, root_node.end_point)]
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens += tree_to_token_index(child)
            if child.type=='comment':
                break
        return code_tokens

def find_type(root_node):

    if (len(root_node.children) == 0 or root_node.type.find('string') != -1):
        return root_node.type
    else:
        code_tokens = []
        for child in root_node.children:
            code_tokens.append(find_type(child))
            if child.type=='comment':
                break
        return code_tokens

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

def get_identifier_names(parser,code):
    # Parse the source code
    tree = parser.parse(bytes(code, "utf8"))
    root_node = tree.root_node

    tokens_index = tree_to_token_index(root_node)
    cpp_loc = code.split('\n')
    code_tokens = [index_to_code_token(x, cpp_loc) for x in tokens_index]
    identifiers = find_type(root_node)
    identifiers = flatten(identifiers)
    identifier_names=[]

    if len(code_tokens) == len(identifiers):
        for i in range(len(code_tokens)):
            if identifiers[i]=='identifier' or identifiers[i]=='name':
                identifier_names.append(code_tokens[i])

    return identifier_names
def main():

    parser = ArgumentP

    parser.add_argument('--input', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--tree', type=str, default=None)
    parser.add_argument('--random_seed', type=int, default=42)
    parser.add_argument('--code_type', type=str, default=None)



    args = parser.parse_args()
    tokenizer = RobertaTokenizer.from_pretrained('Salesforce/codet5-base')
    data=load_data(args.input)
    random.seed(args.random_seed)
    random.shuffle(data)
    sys.setrecursionlimit(3000)

    if args.code_type !='python':
        lang = tree_sitter.Language(args.tree, args.code_type)
        parser = tree_sitter.Parser()
        parser.set_language(lang)

    with open(args.output, 'w') as f:

        for idx, items in enumerate(tqdm(data)):
            replacelist = tokenizer.additional_special_tokens
            replacelist = list(reversed(replacelist))
            if args.code_type =='python':
                if 'code_tokens' in items:
                    code = ' '.join(items['code_tokens'])
                    generate = items['code_tokens']
                else:
                    code = ' '.join(items['function_tokens'])
                    generate = items['function_tokens']
                nl = ' '.join(items['docstring_tokens'])

                tokens = tokenize.tokenize(BytesIO(code.encode('utf-8')).readline)
                try:
                    filtered_tokens = [tok for tok in tokens if
                                       tok.type == tokenize.NAME and not keyword.iskeyword(tok.string)]
                except IndentationError:
                    continue
                except tokenize.TokenError:
                    continue

                Identifiers = []
                for tok in filtered_tokens:
                    Identifiers.append(tok.string)
            else:
                if 'code_tokens' in items:
                    generate = items['code_tokens']
                else:
                    generate = items['function_tokens']
                nl = ' '.join(items['docstring_tokens'])

                code=items['code']
                if args.code_type=='php':
                    code = '<?php ' + code + ' ?>'

                Identifiers=get_identifier_names(parser, code)

                # tree_site can't find php identity, only find name, so we need filter it.
                if args.code_type == 'php':
                    Identifiers = [item for item in Identifiers if item not in php_keylist]
                    if len(Identifiers) == 0:
                        continue
                    new_arr, deleted_elements = delete_elements(Identifiers)
                    Identifiers = new_arr
                    if len(Identifiers) == 0:
                        continue
                else:
                    len_identity = len(Identifiers)
                    if len(Identifiers) == 0:
                        continue
                    if args.code_type=='javascript':
                        ratio=0.1
                    else:
                        ratio=0.5
                    no_delete = random.sample(Identifiers, k=int(max(ratio * (len_identity - 1), 1)))
                    Identifiers = [item for item in Identifiers if item in no_delete]

            indices = []
            for index, item in enumerate(generate):
                if item in Identifiers:
                    indices.append(index)

            num = len(Identifiers)
            idxs = [i for i in range(num)]

            result = []
            my_dict = {}
            for idx,item in enumerate(Identifiers):
                if item not in my_dict and idx in idxs:
                    my_dict[item] = 1
                    result.append(item)

            result = result[:len(replacelist)-1]
            replacelist_new = replacelist[:len(result)]
            mapping=dict(zip(result,replacelist_new))
            label=[]

            for idex,i in enumerate(sorted(idxs)):
                if Identifiers[i] not in mapping:
                    continue
                Identifiers[i] = mapping[Identifiers[i]]

            for i in range(len(result)):
                label.append(replacelist[i])
                label.append(result[i])
            label.append(replacelist[i+1])

            generate_mapping=dict(zip(indices, Identifiers))
            for idx,item in enumerate(generate):
                if idx in generate_mapping:
                    generate[idx]=generate_mapping[idx]

            positive=generate
            positive = ' '.join(positive)
            labels = ' '.join(label)
            group = {}
            positives = []
            positives.append(positive)
            positives = tokenizer(positives, add_special_tokens=False, max_length=240, truncation=True, padding=False)[
                'input_ids']
            query = tokenizer.encode(nl, add_special_tokens=False, max_length=60, truncation=True)
            labels = tokenizer(labels, add_special_tokens=False, max_length=60, truncation=True, padding=False)[
                'input_ids']
            group['query'] = query
            group['positives'] = positives
            group['labels'] = labels
            group['negatives'] = []
            f.write(json.dumps(group) + '\n')

if __name__ == '__main__':
    main()