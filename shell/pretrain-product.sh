export PRETRAIN_PATH=/data1/lixinze/shop/pretrain-data-identity
export pretrain_data=product_train.jsonl

# PRETRAIN_PATH is the path where the pretraining data is located.
# pretrain_data is product pretraining dataset. 

export OUPUT=/data1/lixinze/shop/shop-pretrain
export output_dir=shop_check
export logging_dir=shop_log

# OUPUT is the path to save checkponit and log.
# output_dir is checkpoint path.
# logging_dir is logging path.

cd ../
python train_santa.py \
     --output_dir ${OUPUT}/${output_dir}  \
     --model_name_or_path /data1/lixinze/t5/t5-base  \
     --do_train  \
     --save_steps 1000  \
     --train_path ${PRETRAIN_PATH}/${pretrain_data} \
     --per_device_train_batch_size 16  \
     --train_n_passages 1  \
     --learning_rate 1e-4  \
     --q_max_len 50  \
     --p_max_len 240  \
     --l_max_len 35  \
     --num_train_epochs 6  \
     --use_generate True \
     --logging_dir ${OUPUT}/${logging_dir}
