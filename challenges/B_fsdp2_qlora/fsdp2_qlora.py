import torch
import os
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer, SFTConfig
from datasets import load_dataset

# Challenge B: QLoRA + FSDP2
# Goal: Loss equivalence on 2xT4 GPUs

def setup_fsdp_qlora(model_name="unsloth/llama-3-8b-bnb-4bit"):
    # 1. BitsAndBytes Config
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    
    # 2. Load Model
    # Note: For FSDP2, we usually load on CPU or use meta device
    # but the puzzle asks for transformers/Trainer compatibility
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map={"": int(os.environ.get("LOCAL_RANK", 0))}, # Map to local rank
        torch_dtype=torch.bfloat16,
    )
    
    # 3. Prepare for kbit training
    model = prepare_model_for_kbit_training(model)
    
    # 4. LoRA Config
    lora_config = LoraConfig(
        r=64,
        lora_alpha=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    return model

# Kaggle execution logic for B_fsdp2.ipynb will use this pattern
