#!/bin/bash

# python eval/perplexity.py -m meta-llama/Llama-2-7b-chat-hf --dataset pg19 --split test --feature text --save-tokenized output/pg19-test-tokenized
#PG19="--tokenized emozilla/pg19-test-tokenized"

# python eval/perplexity.py -m meta-llama/Llama-2-7b-chat-hf --dataset tau/scrolls --subset gov_report --split test --feature input --save-tokenized output/govreport-test-tokenized
#GOVREPORT="--tokenized emozilla/govreport-test-tokenized --dataset-min-tokens 16384 --samples 50"

# python eval/perplexity.py -m meta-llama/Llama-2-7b-chat-hf --dataset hoskinson-center/proof-pile --split test --feature text --save-tokenized output/proofpile-test-tokenized

PROOFPILE_LONG_SMALL="--tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized --dataset-min-tokens 131072 --samples 10 --truncate"


PROOFPILE_LONG_SMALL_MISTRAL="--tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-mistral --dataset-min-tokens 131072 --samples 10 --truncate --split train"

PROOFPILE_LONG_SMALL_OLMO="--tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-olmo --dataset-min-tokens 131072 --samples 10 --truncate --split train"

PROOFPILE_LONG_SMALL_LLAMA3_1="--tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-llama3.1 --dataset-min-tokens 131072 --samples 10 --truncate --split train"

PROOFPILE_LONG_SMALL_QWEN2_5="--tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-qwen2.5 --dataset-min-tokens 131072 --samples 10 --truncate --split train"

CUSTOM="--custom-model-together"

# 以下展示了一套在全新数据集上进行评估的代码，以供参考

# 首先，使用以下命令生成数据集的tokenized版本，将其保存在本地（save-tokenized指定本地路径）
#python eval/perplexity.py  -m meta-llama/Llama-3.1-8B-Instruct --dataset /data/qytian/hw/radixyarn/testset/proofpile-test --split test --feature text --save-tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-llama3.1

#python eval/perplexity.py  -m Qwen/Qwen2.5-1.5B-Instruct --dataset /data/qytian/hw/radixyarn/testset/proofpile-test --split test --feature text --save-tokenized /data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-qwen2.5

# 其次使用以下命令进行评估，--yarn表示使用原本yarn方法，--radix表示使用改进方法，后面的参数表示扩展上下文的倍数，min-tokens表示测试时最小token数，max-tokens表示测试时最大token数，tokens-step表示每次扩展的token数，aggressive-memory表示每测试完一组token长度就删一次显存（以免爆显存），--original-max-position-embeddings表示原始模型的最大位置嵌入数
# python eval/perplexity.py \
#     ${PROOFPILE_LONG_SMALL_LLAMA3_1} \
#     --output-file output/proofpile-llama3-2-mrrope.csv \
#     --original-max-position-embeddings 8192 \
#     --min-tokens 8192 --max-tokens 32768 --tokens-step 2048 --aggressive-memory \
#     --radix 4 \
#     -m meta-llama/Llama-3.2-3B-Instruct

# 18,35
# python eval/perplexity.py \
#     ${PROOFPILE_LONG_SMALL_LLAMA3_1} \
#     --output-file output/proofpile-llama3-2-yarn.csv \
#     --original-max-position-embeddings 8192 \
#     --min-tokens 8192 --max-tokens 32768 --tokens-step 2048 --aggressive-memory \
#     --radix 10 \
#     -m meta-llama/Llama-3.2-3B-Instruct
    


# python eval/perplexity.py \
#    ${PROOFPILE_LONG_SMALL} \
#    --output-file output/r2-llama2-7b.csv \
#    --original-max-position-embeddings 4096 \
#    --min-tokens 4096 --max-tokens 40960 --tokens-step 2048 --aggressive-memory \
#    --radix 10 \
#    -m meta-llama/Llama-2-7b-chat-hf 


#python eval/perplexity.py \
#    ${PROOFPILE_LONG_SMALL_OLMO} \
#    --output-file output/proofpile-long-small-olmo-yarn.csv \
#    --original-max-position-embeddings 4096 \
#    --min-tokens 4096 --max-tokens 40960 --tokens-step 2048 --aggressive-memory \
#    --yarn 10 \
#    -m /data/models_other/OLMo-1B-0724-hf/main


# python eval/perplexity.py \
#     ${PROOFPILE_LONG_SMALL_QWEN2_5} \
#     --output-file output/pp-qwen2.5-radix-8bit.csv \
#     --original-max-position-embeddings 32768 \
#     --min-tokens 8192 --max-tokens 131072 --tokens-step 8192 --aggressive-memory --load-in-8bit \
#     --radix 4 \
#     -m Qwen/Qwen2.5-3B-Instruct

# baseline in 8192= 3.3132457733154297, 2.82656621932983, 2.6006536,2.489299, 2.4130609,2.355232
# now in 8192= 3.3152356147766113, 2.8283824920654207
# python eval/perplexity.py \
#     ${PROOFPILE_LONG_SMALL_MISTRAL} \
#     --output-file output/r3-mistral7bv2-32K-48K.csv \
#     --original-max-position-embeddings 32768 \
#     --min-tokens 32768 --max-tokens 49152 --tokens-step 8192 --aggressive-memory --flash-attention \
#     --radix 4 \
#     -m mistralai/Mistral-7B-Instruct-v0.2


# 3.646873950958252, 3.40777564048767, 3.04238152503967, 2.84393000602722
# python eval/perplexity.py \
#     ${PROOFPILE_LONG_SMALL_LLAMA3_1} \
#     --output-file output/yarn-llama38b.csv \
#     --original-max-position-embeddings 8192 \
#     --min-tokens 4096 --max-tokens 32768 --tokens-step 4096 --aggressive-memory \
#     --yarn 4 \
#     -m meta-llama/Llama-3.1-8B-Instruct


python eval/perplexity.py \
    ${PROOFPILE_LONG_SMALL_LLAMA3_1} \
    --output-file output/mrrope-llama38b.csv \
    --original-max-position-embeddings 8192 \
    --min-tokens 4096 --max-tokens 32768 --tokens-step 4096 --aggressive-memory \
    --radix 4 \
    -m meta-llama/Llama-3.1-8B-Instruct
