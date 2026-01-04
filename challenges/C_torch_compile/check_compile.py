import torch
import torch._dynamo
from transformers import AutoModelForCausalLM, BitsAndBytesConfig
from peft import LoraConfig, get_peft_model

# Challenge C: torch.compile without graph breaks
# Goal: 0 graph breaks, < 30 recompilations

def check_graph_breaks(model_name="unsloth/llama-3-8b-bnb-4bit"):
    # 1. Setup Model
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16, # torch.compile likes fp16/bf16
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
    )
    
    lora_config = LoraConfig(
        r=8,
        target_modules=["q_proj", "v_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    
    # 2. Compile Check
    # We use dynamo.explain to see what's causing breaks
    def forward_step(input_ids, labels):
        outputs = model(input_ids=input_ids, labels=labels)
        return outputs.loss

    # Mock data
    input_ids = torch.randint(0, 32000, (1, 128)).cuda()
    labels = input_ids.clone()

    print("Checking for graph breaks...")
    explanation = torch._dynamo.explain(forward_step, input_ids, labels)
    print(explanation)
    
    # Common fixes for graph breaks in QLoRA:
    # - torch._dynamo.allow_in_graph(custom_op)
    # - Avoiding .item() in forward
    # - Using torch.where instead of python if
    
if __name__ == "__main__":
    # This will be run on Kaggle T4
    pass
