export CHECK_POINT=/data1/lixinze/Adv/finetune/adv-SANTA-128/SANTA-2e-5_128
export EVAL_PATH=/data1/lixinze/Adv
export eval_data_file=valid.jsonl
export codebase_file=valid.jsonl

# CHECK_POINT is the directory which includes checkponits

cd ../code_best_dev
python evaluate_code_finetune.py \
    --model_name_or_path ${CHECK_POINT} \
    --eval_data_file ${EVAL_PATH}/${eval_data_file}  \
    --codebase_file ${EVAL_PATH}/${codebase_file}  \
    --code_length 240 \
    --nl_length 50 \
    --eval_batch_size 64


