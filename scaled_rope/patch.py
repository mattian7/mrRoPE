import torch

# Yarn-method        
def patch_llama_for_yarn_scaled_rotary_embeddings(model, head_dim, scale, base, max_position_embeddings,original_max_position_embeddings):
    from .LlamaYaRNScaledRotaryEmbedding import LlamaYaRNScaledRotaryEmbedding
    model.model.rotary_emb = LlamaYaRNScaledRotaryEmbedding(
        head_dim, max_position_embeddings=max_position_embeddings, scale=scale, base=base, 
        original_max_position_embeddings=original_max_position_embeddings, 
        device=model.model.rotary_emb.inv_freq.device)
    

# Our-method

def patch_llama_for_yarn_radix_embeddings(model, head_dim, scale, base, max_position_embeddings,original_max_position_embeddings):
    from .LlamaYaRNRadix import LlamaYaRNRadix
    model.model.rotary_emb = LlamaYaRNRadix(
        head_dim, max_position_embeddings=max_position_embeddings, scale=scale, base=base, 
        original_max_position_embeddings=original_max_position_embeddings, 
        device=model.model.rotary_emb.inv_freq.device)