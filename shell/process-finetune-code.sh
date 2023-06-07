export FINETUNE_RAW_PATH=/data1/lixinze/Adv
export finetune_raw_data=train.jsonl
export FINETUNE_PATH=/data1/lixinze/Adv/finetune
export finetune_data=adv_train.jsonl

cd ../processing/Code
python build_code.py \
--input ${FINETUNE_RAW_PATH}/${finetune_raw_data} \
--output ${FINETUNE_PATH}/${finetune_data}