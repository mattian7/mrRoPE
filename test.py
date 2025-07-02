from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

model_name = "Qwen/Qwen3-0.6B"

# 下载 tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name)

# 下载并加载模型
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")

# 测试推理
inputs = tokenizer("Hello, who are you?", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=50)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))