from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
from huggingface_hub import login
import datasets
import torch

input_texts = datasets.load_from_disk("/data/qytian/hw/radixyarn/testset/proofpile-test-tokenized-qwen2.5")
input_texts = input_texts.filter(lambda x: x["tokenized_len"] >= 32768)
input_text = input_texts[0:1]

model_name = "Qwen/Qwen2.5-0.5B-Instruct"

# 下载 tokenizer
#tokenizer = AutoTokenizer.from_pretrained(model_name)

# 下载并加载模型
config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            load_in_8bit=False,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
torch_dtype = None
config.pretraining_tp = 1

model = AutoModelForCausalLM.from_pretrained(
    model_name, device_map="balanced_low_0", config=config,
    quantization_config=quantization_config, use_flash_attention_2=True)

input_ids = torch.tensor(input_text["input_ids"]).to(model.device)
target_ids = input_ids.clone()
# 测试推理
#inputs = tokenizer("Hello, who are you?", return_tensors="pt").to(model.device)
#outputs = model.generate(**input_text, max_new_tokens=1)
#print(tokenizer.decode(outputs[0], skip_special_tokens=True))

with torch.no_grad():
    outputs = model(input_ids, labels=target_ids)
    neg_log_likelihood = outputs.loss

print(neg_log_likelihood)   

