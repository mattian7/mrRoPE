#!/bin/bash

# python eval/perplexity.py -m meta-llama/Llama-2-7b-hf --dataset pg19 --split test --feature text --save-tokenized output/pg19-test-tokenized
#PG19="--tokenized emozilla/pg19-test-tokenized"

# python eval/perplexity.py -m meta-llama/Llama-2-7b-hf --dataset tau/scrolls --subset gov_report --split test --feature input --save-tokenized output/govreport-test-tokenized
#GOVREPORT="--tokenized emozilla/govreport-test-tokenized --dataset-min-tokens 16384 --samples 50"

# python eval/perplexity.py -m meta-llama/Llama-2-7b-hf --dataset hoskinson-center/proof-pile --split test --feature text --save-tokenized output/proofpile-test-tokenized

PROOFPILE_LONG_SMALL="--tokenized /data/qytian/hw/yarn/testset/proofpile-test-tokenized --dataset-min-tokens 131072 --samples 10 --truncate"


PROOFPILE_LONG_SMALL_MISTRAL="--tokenized /data/qytian/hw/yarn/testset/proofpile-test-tokenized-mistral --dataset-min-tokens 131072 --samples 10 --truncate --split train"

PROOFPILE_LONG_SMALL_OLMO="--tokenized /data/qytian/hw/yarn/testset/proofpile-test-tokenized-olmo --dataset-min-tokens 131072 --samples 10 --truncate --split train"

CUSTOM="--custom-model-together"

# 以下展示了一套在全新数据集上进行评估的代码，以供参考

# 首先，使用以下命令生成数据集的tokenized版本，将其保存在本地（save-tokenized指定本地路径）
# python eval/perplexity.py  -m meta-llama/Llama-2-7b-hf --dataset hoskinson-center/proof-pile --split test --feature text --save-tokenized /data/qytian/hw/yarn/testset/proofpile-test-tokenized

# 其次使用以下命令进行评估，--yarn表示使用原本yarn方法，--radix表示使用改进方法，后面的参数表示扩展上下文的倍数，min-tokens表示测试时最小token数，max-tokens表示测试时最大token数，tokens-step表示每次扩展的token数，aggressive-memory表示每测试完一组token长度就删一次显存（以免爆显存），--original-max-position-embeddings表示原始模型的最大位置嵌入数
#python eval/perplexity.py \
#    ${PROOFPILE_LONG_SMALL} \
#    --output-file output/proofpile-long-small-yarn.csv \
#    --original-max-position-embeddings 4096 \
#    --min-tokens 4096 --max-tokens 40960 --tokens-step 2048 --aggressive-memory \
#    --yarn 10 \
#    -m meta-llama/Llama-2-7b-chat-hf 
    


python eval/perplexity.py \
    ${PROOFPILE_LONG_SMALL_OLMO} \
    --output-file output/proofpile-long-small-olmo-radix.csv \
    --original-max-position-embeddings 4096 \
    --min-tokens 4096 --max-tokens 40960 --tokens-step 2048 --aggressive-memory \
    --radix 10 \
    -m /data/models_other/OLMo-1B-0724-hf/main


python eval/perplexity.py \
    ${PROOFPILE_LONG_SMALL_OLMO} \
    --output-file output/proofpile-long-small-olmo-yarn.csv \
    --original-max-position-embeddings 4096 \
    --min-tokens 4096 --max-tokens 40960 --tokens-step 2048 --aggressive-memory \
    --yarn 10 \
    -m /data/models_other/OLMo-1B-0724-hf/main