import argparse
import datasets
import gc
import sys
import numpy as np
import matplotlib.pyplot as plt

import torch
import warnings
from transformers import AutoTokenizer
from tqdm import tqdm
from eval.model_loader import *
from huggingface_hub import login


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def get_yarn_freq(low, high,S):
    pos = range(64)
    base = 10000 ** (-1 / 64)
    low = 20
    high = 46
    freqs = []
    for i in pos:
        if i <= low:
            freqs.append(base**i)
        elif i >= high:
            scale =S
            freqs.append(1/scale * (base**i))
        else:
            r = (i - low) / (high - low)
            scale = (1-r) + r/S
            freqs.append(scale * (base**i))
    return np.array(freqs)

def get_radix_freq(low, high,S):
    pos = range(64)
    base = 10000 ** (-1 / 64)
    freqs = []
    scale = S ** (1 / (high - low ))
    for i in pos:
        if i <= low:
            freqs.append(base**i)
        elif i >= high:
            freqs.append((base**i)/S)
        else:
            r = (i - low)
            freqs.append((base**i)/ (scale** r))
    return np.array(freqs)


def get_freq(PE_name, device):

    if PE_name == "RoPE":
        pos_freqs = 10000 ** (torch.arange(0, 128, 2).float().to(device) / 128)
        inv_freq = 1.0 / pos_freqs
        t = torch.arange(8194, device=device)
        freqs = torch.einsum("i,j->ij", t, inv_freq).to(device)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        cos = emb.cos()
        sin = emb.sin()
        print(f"RoPE freq 25: {freqs[100, 25].to(torch.float32)}")
        return cos, sin
    elif PE_name == "YaRN":
        low, high = 20, 46
        S = 10
        yarn_freqs = get_yarn_freq(low, high, S)
        yarn_freqs = torch.tensor(yarn_freqs, device=device)
        t = torch.arange(8194, device=device)
        freqs = torch.einsum("i,j->ij", t, yarn_freqs).to(device)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        cos = emb.cos()
        sin = emb.sin()
        print(f"YaRN freq 25: {freqs[100, 25].to(torch.float32)}")
        return cos, sin
    elif PE_name == "Radix":
        low, high = 20, 46
        S = 10
        radix_freqs = get_radix_freq(low, high, S)
        radix_freqs = torch.tensor(radix_freqs, device=device)
        t = torch.arange(8194, device=device)
        freqs = torch.einsum("i,j->ij", t, radix_freqs).to(device)
        emb = torch.cat((freqs, freqs), dim=-1).to(device)
        cos = emb.cos()
        sin = emb.sin()
        print(f"Radix freq 25: {freqs[100, 25].to(torch.float32)}")
        return cos, sin
    

def calculate_qk_product(Q,K,cos,sin,outputfile):
    q = Q * cos.unsqueeze(0) + rotate_half(Q) * sin.unsqueeze(0)
    k = K * cos.unsqueeze(0) + rotate_half(K) * sin.unsqueeze(0)
    
    qk_product = torch.matmul(q, k.transpose(-1, -2))  

    qk_value = qk_product[0, 1, 2:].cpu().detach().numpy()

    with open(outputfile, "w") as f:
        for value in qk_value:
            f.write(f"{value}\n")
        '''
        f.write(f"q[1]: {Q[0, 1, :].to(torch.float32).cpu().detach().numpy()}\n")
        f.write(f"k[2]: {K[0, 2, :].to(torch.float32).cpu().detach().numpy()}\n")
        f.write(f"k[3]: {K[0, 3, :].to(torch.float32).cpu().detach().numpy()}\n")
        f.write(f"freqs[1]: {cos[1, :].to(torch.float32).cpu().detach().numpy()}\n")
        f.write(f"freqs[2]: {cos[100, :].to(torch.float32).cpu().detach().numpy()}\n")
        '''

def calculate_qk_no_rotary(Q,K, outputfile):
    q = Q[0,1,:].to(torch.float32).cpu().detach().numpy()  # 取Q[0,1,:]，并增加一个维度
    k = K[0,2,:].to(torch.float32).cpu().detach().numpy()
    # 计算qk_products[d] = q[2d] * k[2d] + q[2d+1] * k[2d+1] 
    qk_products = []
    for d in range(64):
        qk_product = q[2*d] * k[2*d] + q[2*d+1] * k[2*d+1]
        qk_products.append(qk_product)
    qk_value = np.array(qk_products)

    with open(outputfile, "w") as f:
        for value in qk_value:
            f.write(f"{value}\n")

    


def main(args):
    model = args.model
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token


    #manual_input_ids = [1, 1369] + [1095] * 8192
    manual_input_ids = [1,  515] + [304] * 8192
    manual_attention_mask = [1] * len(manual_input_ids)
    manual_encoded = {
        "input_ids": torch.tensor([manual_input_ids]),
        "attention_mask": torch.tensor([manual_attention_mask])
    }


    model = load_model(model, args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    outputs = model(**manual_encoded, output_hidden_states=True)
    hidden_states = outputs.hidden_states
    first_layer_hidden_states = hidden_states[0].to(device)
    print(f"hidden_states shape: {len(hidden_states)}")
    q_porj = model.model.layers[0].self_attn.q_proj.weight.to(device)
    k_porj = model.model.layers[0].self_attn.k_proj.weight.to(device)
    '''
    # 获取first_layer_hidden_states在 Q 和 K 的投影结果
    Q = torch.matmul(first_layer_hidden_states, q_porj.T)[:,:,2048:2176]
    K = torch.matmul(first_layer_hidden_states, k_porj.T)[:,:,2048:2176]
    
    cos_o, sin_o = get_freq("RoPE", device)
    cos_y, sin_y = get_freq("YaRN", device)
    cos_r, sin_r = get_freq("Radix", device)
    
    calculate_qk_product(Q, K, cos_o, sin_o, "./analysis/qk_product.txt")
    calculate_qk_product(Q, K, cos_y, sin_y, "./analysis/qk_product_yarn.txt")
    calculate_qk_product(Q, K, cos_r, sin_r, "./analysis/qk_product_radix.txt")
    '''

    Q = torch.matmul(first_layer_hidden_states, q_porj.T)
    K = torch.matmul(first_layer_hidden_states, k_porj.T)

    calculate_qk_no_rotary(Q[:,:,:128], K[:,:,:128], "./analysis/value_0.txt")
    calculate_qk_no_rotary(Q[:,:,128:256], K[:,:,128:256], "./analysis/value_1.txt")
    calculate_qk_no_rotary(Q[:,:,2048:2176], K[:,:,2048:2176], "./analysis/value_middle.txt")
    calculate_qk_no_rotary(Q[:,:,3968:], K[:,:,3968:], "./analysis/value_last.txt")


def draw(value):
    # 横坐标为
    pos = range(1,8193)
    plt.plot(pos, value, label='RoPE', color='#318aca', marker='o', linestyle='-')
    plt.xlabel("Distance")
    plt.ylabel("QK value")
    plt.title("QK Product Values across relative distance")
    plt.savefig("./analysis/qk_product.png")



if __name__ == "__main__":
    warnings.simplefilter("ignore")
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="meta-llama/Llama-2-7b-chat-hf")
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--min-tokens", type=int, default=4096)
    main(add_args(parser).parse_args())
