from argparse import ArgumentParser
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from scaled_rope.patch import *
from time import sleep

def load_model(model, args):
    # 先读取模型的config,根据args把要使用的index外推方法注入
    # factor就是scale
    if args.custom_model:
        from scaled_rope.modeling_llama_yarn import LlamaForCausalLM
        from scaled_rope.configuration_llama import LlamaConfig
        model_cls = LlamaForCausalLM
        config_cls = LlamaConfig
    elif args.custom_model_together:
        from scaled_rope.modeling_llama_together_yarn import LlamaForCausalLM
        from scaled_rope.configuration_llama import LlamaConfig
        model_cls = LlamaForCausalLM
        config_cls = LlamaConfig
    elif args.custom_model_mistral:
        from scaled_rope.modeling_mistral_yarn import MistralForCausalLM
        from scaled_rope.configuration_mistral import MistralConfig
        model_cls = MistralForCausalLM
        config_cls = MistralConfig
    else:
        model_cls = AutoModelForCausalLM
        config_cls = AutoConfig

    config = config_cls.from_pretrained(
        model, trust_remote_code=not args.custom_model)
    
    args.head_dim = config.hidden_size // config.num_attention_heads
    if args.max_position_embeddings:
        config.max_position_embeddings = args.max_position_embeddings
    if args.factor:
        config.rope_scaling["factor"] = args.factor
    if args.no_use_cache:
        config.use_cache = False
    else:
        config.use_cache = True
    if args.sliding_window_attention:
        config.sliding_window = args.sliding_window_attention
    if args.custom_model or args.custom_model_together or args.custom_model_mistral:
        if args.radix:
            config.rope_scaling = {
                "type": "radix",
                "factor": args.radix,
                "original_max_position_embeddings": args.original_max_position_embeddings,
            }
        elif args.yarn:
            config.rope_scaling = {
                "type": "yarn",
                "factor": args.yarn,
                "original_max_position_embeddings": args.original_max_position_embeddings,
            }

    if args.load_in_8bit or args.load_in_4bit:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=args.load_in_4bit,
            load_in_8bit=args.load_in_8bit,
            llm_int8_threshold=6.0,
            llm_int8_has_fp16_weight=False,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        torch_dtype = None
        config.pretraining_tp = 1
    else:
        quantization_config = None
        torch_dtype = torch.bfloat16

    loaded = model_cls.from_pretrained(
        model,
        torch_dtype=torch_dtype,
        device_map="auto",
        trust_remote_code=not args.custom_model,
        config=config,
        quantization_config=quantization_config,
        use_flash_attention_2=args.flash_attention,
    )
    return loaded


def add_args(parser: ArgumentParser):
    parser.add_argument("--yarn", type=float)
    parser.add_argument("--radix", type=float)
    parser.add_argument("--factor", type=float)
    parser.add_argument("--load-in-8bit", action="store_true")
    parser.add_argument("--load-in-4bit", action="store_true")
    parser.add_argument("--finetuned", action="store_true")
    parser.add_argument("--gpt-neox-max-length", type=int)
    parser.add_argument("--adapter", type=str)
    parser.add_argument("--max-position-embeddings", type=int)
    parser.add_argument("--original-max-position-embeddings", type=int)
    parser.add_argument("--sliding-window-attention", type=int)
    parser.add_argument("--custom-model", action="store_true")
    parser.add_argument("--custom-model-together", action="store_true")
    parser.add_argument("--custom-model-mistral", action="store_true")
    parser.add_argument("--flash-attention", action="store_true")
    parser.add_argument("--no-use-cache", action="store_true")
    return parser


def apply_patches(model, args):
    if not args.custom_model and not args.custom_model_together and not args.custom_model_mistral:
        if args.yarn:
            patch_llama_for_yarn_scaled_rotary_embeddings(
                model, args.head_dim, scale=args.yarn, 
                max_position_embeddings = int(args.original_max_position_embeddings*args.yarn),original_max_position_embeddings=args.original_max_position_embeddings)
        elif args.radix:
            patch_llama_for_yarn_radix_embeddings(
                model, args.head_dim, scale=args.radix, 
                max_position_embeddings = int(args.original_max_position_embeddings*args.radix),original_max_position_embeddings=args.original_max_position_embeddings)

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter)
        model = model.merge_and_unload()

    return model


def load_model_and_apply_patches(model, args):
    return apply_patches(load_model(model, args), args)
