export TREC_PATH=/data1/lixinze/Adv/test/adv_inference.trec


cd ../evaluate_code
python evaluate_code.py \
    --trec_save_path ${TREC_PATH}  


