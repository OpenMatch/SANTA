export PRETRAIN_RAW_PATH=/data1/lixinze/dataset/ruby/final/jsonl/train
export pretrain_raw_data=ruby_pretrain_no.jsonl

# PRETRAIN_RAW_PATH is the path where the raw pretraining data is located. 
# "pretrain_raw_data" refers to the merged CodeSearchNet data, with separate versions available for each of the six programming languages. 

export PRETRAIN_PATH=/data1/lixinze/dataset/ruby/pretrain_data
export pretrain_data=ruby.jsonl
export language=ruby

# The "language" refers to the names of the six programming languages in CodeSearchNet.
cd ../processing/Code
python build_code_entity.py \
--input ${PRETRAIN_RAW_PATH}/${pretrain_raw_data} \
--tree ./tree-sitter-languages.so \
--output ${PRETRAIN_PATH}/${pretrain_data} \
--code_type ${language}
