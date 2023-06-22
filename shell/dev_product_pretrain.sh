export PRETRAIN_PATH=/data1/lixinze/shop/pretrain-data-identity
export pretrain_eval_data=product_eval.jsonl
export CHECK_POINT=/data1/lixinze/shop/finetune-checkpoint/hard-neg/top-100/t5-shop-1e-5_60
# CHECK_POINT is the directory which includes checkponits

cd ../product_best_dev
python evaluate_shop_pretrain.py \
      --per_gpu_batch_size 64 \
      --test_data ${PRETRAIN_PATH}/${pretrain_eval_data} \
      --block_size 240 \
      --query_block_size 50 \
      --model_path ${CHECK_POINT} \
      --tensorboard_path ${CHECK_POINT}  


