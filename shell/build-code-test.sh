export TEST_PATH=/data1/lixinze/Adv
export test_data=test.jsonl

# TEST_PATH is the path where the raw test data is located.

export OUTPUT_PATH=/data1/lixinze/Adv/test
export output_query=adv_test_query.jsonl
export output_doc=adv_test_doc.jsonl

# OUTPUT_PATH is the path where the processed test data is saved.
# data_type chooses whether to process the query or code.

cd ../evaluate_code
python build_test_code.py \
     --input ${TEST_PATH}/${test_data}  \
     --output ${OUTPUT_PATH}/${output_query}  \
     --data_type query  \

python build_test_code.py \
     --input ${TEST_PATH}/${test_data}  \
     --output ${OUTPUT_PATH}/${output_doc}  \
     --data_type doc  \

