export OUTPUT_PATH=/data1/lixinze/Adv/test/
export MODEL_PATH=/data1/lixinze/Adv/pretrain-checkpoint/cl-identity/best
export DOC_PATH=/data1/lixinze/Adv/test/adv_test_doc.jsonl
export QUERY_PATH=/data1/lixinze/Adv/test/adv_test_query.jsonl
export TREC_PATH=/data1/lixinze/Adv/test/adv_inference.trec

# OUTPUT_PATH is the path where code embeddings are saved.
# MODEL_PATH checkpint
# DOC_PATH is the processed test data

cd ../OpenMatch/src
python -m openmatch.driver.build_index  \
    --output_dir ${OUTPUT_PATH} \
    --model_name_or_path ${MODEL_PATH}  \
    --per_device_eval_batch_size 256  \
    --corpus_path ${DOC_PATH}  \
    --doc_template "<code>"  \
    --q_max_len 50  \
    --p_max_len 240  \
    --dataloader_num_workers 1

python -m openmatch.driver.retrieve  \
    --output_dir ${OUTPUT_PATH}  \
    --model_name_or_path ${MODEL_PATH}  \
    --per_device_eval_batch_size 256  \
    --query_path ${QUERY_PATH} \
    --query_template "<query>"  \
    --q_max_len 50  \
    --trec_save_path ${TREC_PATH}


