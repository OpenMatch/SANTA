export PRETRAIN_RAW_PATH=/data1/lixinze/shop/esci_data/esci_data

# PRETRAIN_RAW_PATH is the path where the raw pretraining data is located.
# PRETRAIN_RAW_PATH includes two file shopping_queries_dataset_examples.parquet and shopping_queries_dataset_products.parquet

export PRETRAIN_PATH=/data1/lixinze/shop/pretrain-data-identity
export pretrain_data=product_train.jsonl
export pretrain_eval_data=product_eval.jsonl

# pretrain_data and pretrain_eval_data are splited from PRETRAIN_RAW_PATH's two files. 
# pretrain_eval_data is used to select best dev checkpoint during pretraining.

cd ../processing/Product
python build_product_entity.py \
--input ${PRETRAIN_RAW_PATH} \
--pretrain_train ${PRETRAIN_PATH}/${pretrain_data} \
--pretrain_eval ${PRETRAIN_PATH}/${pretrain_eval_data} \
