export FINETUNE_RAW_PATH=/data1/lixinze/shop/esci_data/esci_data

# FINETUNE_RAW_PATH is the path where the raw finetuning data is located.
# FINETUNE_RAW_PATH includes two file shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet

export FINETUNE_PATH=/data1/lixinze/shop/finetune-data
export finetune_data=product_train.jsonl
export finetune_eval_data=product_eval.jsonl

# finetune_data and finetune_eval_data are splited from FINETUNE_RAW_PATH's two files. 
# finetune_eval_data is used to select best dev checkpoint during finetuning.

cd ../processing/Product
python build_product.py \
--input ${FINETUNE_RAW_PATH} \
--finetune_train ${FINETUNE_PATH}/${finetune_data} \
--finetune_eval ${FINETUNE_PATH}/${finetune_eval_data} \
