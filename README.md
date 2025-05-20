# mrRoPE

This repo contains the code and data for the YaRN-radix context window extension method.


## Models

### Llama-2-7b-hf

Llama-2-7b-hf has been tested perplexity performance on proofpile-test dataset. The result have been saved into `./output/perplexity_llama`, which shown a better performance of YaRN-radix(YaRN-NTK) compared to YaRN.

In addition, YaRN-radix performance on long context benchmark is also better than YaRN. The preliminary verification on Longbench show results as follows：
| Method | Scale-up time | benchmark   | accuracy   |
| ---: | ------: | :----- | :----- |
|   YaRN |     10 | Longbench | 0.151 |
|   YaRN-radix |    10 | Longbench | **0.176** | 
|   YaRN |     16 | Longbench | 0.143 |
|   YaRN-radix |    16 | Longbench | **0.167** | 


### OLMo-1B-0724-hf

Llama-2-7b-hf has been tested perplexity performance on proofpile-test dataset. The result have been saved into `./output/perplexity_olmo`, which shown a better performance of YaRN-radix(YaRN-NTK) compared to YaRN.

## Reproduction

To reproduce, clone the repository and perform a local installation.

```python
git clone https://github.com/mattian7/mrRoPE
cd mrRoPE
conda create -n mrrope python=3.9
conda activate mrrope
pip install -e .
```


### Evaluation

We prepare to kinds of Scripts for evaluation. Please read the comment in scripts and run for other test-sets.
First line: Evaluation for perplexity test.
Second line: Evaluation for other benchmark (LongBench etc.).

```sh
# ./eval2.sh
# ./evalb.sh
```

### To-do List

1. Test Perplexity on other testsets: gov_report, pg19
2. Test on other long context benchmark: L-EVAL, LooGLE
