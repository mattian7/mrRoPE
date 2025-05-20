#!/bin/bash


# 这份代码是针对longbench数据集的评估脚本，由于longbench本身包括short、medium和long三种长度的文本，因此我们可以通过--filter-length参数来初步选择评估的文本长度，然后再用max-tokens来二次筛选，最终保证整体参与评估的数据最大token数不超过max-tokens；
# 如果需要在其他测试集上测试，请先参考longbench.py撰写新的xx.py，然后进行测试。如果其他数据集没有short、medium和long的划分，可以直接使用max-tokens和min-tokens来筛选数据范围

python eval/longbench.py \
    --original-max-position-embeddings 4096 \
    --yarn 10 \
    --max-tokens 40960 \
    --split train \
    --output-file ./output/longbench-llama-yarn.csv \
    --filter-length short \
    -m meta-llama/Llama-2-7b-chat-hf 

#python eval/longbench.py \
#    --original-max-position-embeddings 4096 \
#    --yarn 16 \
#    --max-tokens 65536 \
#    --min-tokens 40960 \
#    --split train \
#    --output-file ./output/longbench-llama-test2.csv \
#    --filter-length short \
#    --aggressive-memory \
#    -m meta-llama/Llama-2-7b-chat-hf 

