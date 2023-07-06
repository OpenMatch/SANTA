export FINETUNE_PATH=/data1/lixinze/dataset/ruby/finetune_data
export finetune_data=ruby.jsonl
export Pretrain_checkpoint=/data1/lixinze/dataset/ruby/pretrain-checkpoint/ruby

# FINETUNE_PATH is the path where the finetuning data is located.
# finetune_data is one of the six programming language fintuning datasets. 

export OUPUT=/data1/lixinze/dataset/ruby/finetune-checkpoint/ruby
export output_dir=ruby_check
export logging_dir=ruby_log

# OUPUT is the path to save checkponit and log.
# output_dir is checkpoint path.
# logging_dir is logging path.
# train_batch_size 16 for adv and 128 for CodeSearch

cd ../
python train_santa.py \
     --output_dir ${OUPUT}/${output_dir}  \
     --model_name_or_path ${Pretrain_checkpoint}  \
     --do_train  \
     --save_steps 1000  \
     --train_path ${FINETUNE_PATH}/${finetune_data} \
     --per_device_train_batch_size 128  \
     --train_n_passages 1  \
     --learning_rate 1e-5 or 2e-5  \
     --q_max_len 50  \
     --p_max_len 240  \
     --l_max_len 35  \
     --num_train_epochs 10  \
     --use_generate False \
     --logging_dir ${OUPUT}/${logging_dir}
