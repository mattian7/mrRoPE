import torch
import math

# Inverse dim formula to find dim based on number of rotations
def find_correction_dim(num_rotations, dim, base=10000, max_position_embeddings=2048):
    return (dim * math.log(max_position_embeddings/(num_rotations * 2 * math.pi)))/(2 * math.log(base))

# Find dim range bounds based on rotations
# low, high = find_correction_range(self.beta_fast=32, self.beta_slow=1, self.dim, self.base, self.original_max_position_embeddings)
def find_correction_range(low_rot, high_rot, dim, base=10000, max_position_embeddings=2048):
    low = math.floor(find_correction_dim(
        low_rot, dim, base, max_position_embeddings))
    high = math.ceil(find_correction_dim(
        high_rot, dim, base, max_position_embeddings))
    return max(low, 0), min(high, dim-1)  # Clamp values just in case

def linear_ramp_mask(min, max, dim):
    if min == max:
        max += 0.001  # Prevent singularity

    linear_func = (torch.arange(dim, dtype=torch.float32) - min) / (max - min)
    ramp_func = torch.clamp(linear_func, 0, 1)
    return ramp_func

def get_mscale(scale=1):
    if scale <= 1:
        return 1.0
    return 0.1 * math.log(scale) + 1.0

class LlamaYaRNRadix(torch.nn.Module):
    # dim= k_div = q_div = v_div
    def __init__(self, dim, max_position_embeddings=2048, base=10000, scale=1, original_max_position_embeddings=2048, extrapolation_factor=1, attn_factor=1, beta_fast=32, beta_slow=1, finetuned=False, device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.scale = scale
        self.original_max_position_embeddings = original_max_position_embeddings
        self.extrapolation_factor = extrapolation_factor
        self.attn_factor = attn_factor
        self.beta_fast = beta_fast
        self.beta_slow = beta_slow
        
        #self.yarn_radix(device)
        self.yarn_radix3(device)
        # Build here to make `torch.jit.trace` work.
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        dtype = torch.get_default_dtype()

        self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(dtype), persistent=False)
        self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(dtype), persistent=False)
        
    #def forward(self, x, seq_len=None):
    def forward(self, x, position_ids):
        # x: [bs, num_attention_heads, seq_len, head_size]
        # This `if` block is unlikely to be run after we build sin/cos in `__init__`. Keep the logic here just in case.
        seq_len = position_ids.shape[1]
        if seq_len > self.max_seq_len_cached:
            print(">>>>>>>>use if statement<<<<<<<<<<")
            self.max_seq_len_cached = seq_len

            t = torch.arange(self.max_seq_len_cached, device=self.inv_freq.device, dtype=self.inv_freq.dtype)
            freqs = torch.einsum("i,j->ij", t, self.inv_freq)
            # Different from paper, but it uses a different permutation in order to obtain the same calculation
            emb = torch.cat((freqs, freqs), dim=-1).to(x.device)

            self.register_buffer("cos_cached", (emb.cos() * self.mscale).to(x.dtype), persistent=False)
            self.register_buffer("sin_cached", (emb.sin() * self.mscale).to(x.dtype), persistent=False)
        return (
            self.cos_cached[:seq_len, :].unsqueeze(0).expand(x.shape[0], -1, -1).to(dtype=x.dtype).to(x.device),
            self.sin_cached[:seq_len, :].unsqueeze(0).expand(x.shape[0], -1, -1).to(dtype=x.dtype).to(x.device),
        )

    def yarn_radix(self, device):
        print(">>>>>>>>use yarn4() method<<<<<<<<<<")
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq = 1.0 / pos_freqs

        #low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        low, high = find_correction_range(32, 1, self.dim, self.base, self.original_max_position_embeddings)
        
        print(">>>>>>>>>low, high:", low, high)
        half_dim = self.dim // 2
        inv_freq_interpolation_l = self.scale ** (1.0 / (high-low))

        values = []
        inv_freq_new = []
        r = []
        for i in range(self.dim // 2):
            if i < low+1:
                values.append(1.0)
                inv_freq_new.append(inv_freq[i])
                r.append(0.0)
            elif i > high-1:
                values.append(inv_freq_interpolation_l ** (high-low))
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(1.0)
            else:
                ratio = (i-low)/(high-low)
                values.append((inv_freq_interpolation_l ** (i-low) ))
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(ratio)


        inv_freq_scale = torch.tensor(values, device=device)
        inv_freq_new= torch.tensor(inv_freq_new, device=device)
        inv_freq = inv_freq_new

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(get_mscale(self.scale) * self.attn_factor) 


    def yarn_radix2(self, device):
        print(">>>>>>>>use yarn5() method<<<<<<<<<<")
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq = 1.0 / pos_freqs

        #low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        low, high = find_correction_range(32, 1, self.dim, self.base, self.original_max_position_embeddings)
        middle = int((low + high) // 2)
        
        print(">>>>>>>>>low, high,  middle:", low, high, middle)
        half_dim = self.dim // 2
        inv_freq_interpolation_l = self.scale ** (1.0 / (high-low))
        part_ratio = 3/8
        inv_freq_interpolation_l1 = (self.scale ** part_ratio) ** (1 / (middle - low))
        inv_freq_interpolation_l2 = (self.scale ** (1-part_ratio)) ** ( 1 / (high - middle))

        values = []
        inv_freq_new = []
        r = []
        for i in range(self.dim // 2):
            if i < low+1:
                values.append(1.0)
                inv_freq_new.append(inv_freq[i])
                r.append(0.0)
            elif i > high-1:
                values.append(inv_freq_interpolation_l ** (high-low))
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(1.0)
            elif i < middle +1:
                ratio = (i-low)/(middle-low)
                values.append((inv_freq_interpolation_l1 ** (i-low) ))
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(ratio)
            else:
                ratio = (i-middle)/(high-middle)
                values.append((inv_freq_interpolation_l2 ** (i-middle) ))
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(ratio)


        inv_freq_scale = torch.tensor(values, device=device)
        inv_freq_new= torch.tensor(inv_freq_new, device=device)
        inv_freq = inv_freq_new

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(get_mscale(self.scale) * self.attn_factor) 

    def yarn_radix3(self, device):
        print(">>>>>>>>use yarn6() method<<<<<<<<<<")
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq = 1.0 / pos_freqs

        
        low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        
        
        print(">>>>>>>>>low, high:", low, high)
        half_dim = self.dim // 2
        inv_freq_interpolation_d = self.scale ** (1.0 / ((high-low)*(high-low+1)))

        values = []
        inv_freq_new = []
        r = []
        for i in range(self.dim // 2):
            if i < low+1:
                values.append(1.0)
                inv_freq_new.append(inv_freq[i])
                r.append(0.0)
            elif i > high-1:
                values.append(self.scale)
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(1.0)
            else:
                ratio = (i+1-low)*(i-low)/((high-low)*(high-low+1))
                values.append(inv_freq_interpolation_d ** ((i+1-low)*(i-low)))
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(ratio)


        inv_freq_scale = torch.tensor(values, device=device)
        inv_freq_new= torch.tensor(inv_freq_new, device=device)
        inv_freq = inv_freq_new

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(get_mscale(self.scale) * self.attn_factor) 

    def yarn_radix4(self, device):
        print(">>>>>>>>use yarn7() method<<<<<<<<<<")
        pos_freqs = self.base ** (torch.arange(0, self.dim, 2).float().to(device) / self.dim)
        inv_freq = 1.0 / pos_freqs

        #low, high = find_correction_range(self.beta_fast, self.beta_slow, self.dim, self.base, self.original_max_position_embeddings)
        low, high = find_correction_range(32, 1, self.dim, self.base, self.original_max_position_embeddings)
        
        
        print(">>>>>>>>>low, high:", low, high)
        half_dim = self.dim // 2
        inv_freq_interpolation_d1 = self.scale ** (1.0 / ((high-low)*(high-low+1)))

        values = []
        inv_freq_new = []
        r = []
        for i in range(self.dim // 2):
            if i < low+1:
                values.append(1.0)
                inv_freq_new.append(inv_freq[i])
                r.append(0.0)
            elif i > high-1:
                values.append(self.scale)
                inv_freq_new.append(inv_freq[i]/values[i])
                r.append(1.0)
            else:
                values.append(inv_freq_interpolation_d1 ** ((2*high+1-i-low)*(i-low)))
                inv_freq_new.append(inv_freq[i]/values[i])
                #r.append(ratio)


        #inv_freq_scale = torch.tensor(values, device=device)
        inv_freq_new= torch.tensor(inv_freq_new, device=device)
        inv_freq = inv_freq_new

        self.register_buffer("inv_freq", inv_freq)
        self.mscale = float(get_mscale(self.scale) * self.attn_factor) 