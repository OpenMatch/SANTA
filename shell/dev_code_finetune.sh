export CHECK_POINT=/data1/lixinze/dataset/ruby/pretrain-checkpoint/ruby-SANTA128-new-0.5/SANTA-5e-5-128
export EVAL_PATH=/data1/lixinze/dataset/ruby/final/jsonl/valid
export eval_data_file=ruby_valid_0.jsonl
export codebase_file=ruby_valid_0.jsonl

# CHECK_POINT is the directory which includes checkponits

cd ../code_best_dev
python evaluate_code_pretrain.py \
    --model_name_or_path ${CHECK_POINT} \
    --eval_data_file ${EVAL_PATH}/${eval_data_file}  \
    --codebase_file ${EVAL_PATH}/${codebase_file}  \
    --code_length 240 \
    --nl_length 50 \
    --eval_batch_size 64


