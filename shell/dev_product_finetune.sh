export FINETUNE_PATH=/data1/lixinze/shop/finetune-data
export finetune_eval_data=product_eval.jsonl
export CHECK_POINT=/data1/lixinze/shop/finetune-checkpoint/hard-neg/top-100/t5-shop-1e-5_60
# CHECK_POINT is the directory which includes checkponits
# batch_size is must 1
cd ../product_best_dev
python evaluate_shop_finetune.py \
      --per_gpu_batch_size 1 \
      --test_data ${FINETUNE_PATH}/${finetune_eval_data} \
      --shop_product 256 \
      --shop_query 20 \
      --model_path ${CHECK_POINT} \
      --tensorboard_path ${CHECK_POINT}  


