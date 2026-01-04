# Unsloth Puzzle Challenges: Technical Documentation

This document provides a detailed explanation and code walk-through for all five Unsloth Puzzle Challenges (A-E).

---

## Challenge A: NF4 Triton Kernel
**Objective**: Implement a custom Triton kernel to dequantize NF4 (NormalFloat 4) weights into BFloat16/Float16, matching the performance and logic of Unsloth's optimized kernels.

### Technical Explanation
NF4 is a specialized 4-bit quantization format. Each byte in the weight tensor contains two 4-bit values (nibbles). The kernel must:
1.  Unpack the byte into low and high nibbles.
2.  Map these nibbles to their corresponding floating-point values using a `code` lookup table (standard NF4 values).
3.  Scale the values using `absmax` constants stored per block of 64 elements.

### Implementation Snippet
```python
@triton.jit
def dequantize_nf4_kernel(weight_ptr, absmax_ptr, code_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    
    # 1 byte = 2 elements (4 bits each)
    byte_offsets = (block_start // 2) + tl.arange(0, BLOCK_SIZE // 2)
    mask = byte_offsets < (n_elements // 2)
    
    packed_weights = tl.load(weight_ptr + byte_offsets, mask=mask)
    
    # Unpack nibbles
    low_nibble = (packed_weights & 0xF).to(tl.int32)
    high_nibble = (packed_weights >> 4).to(tl.int32)
    
    # Map to NF4 values
    val_low = tl.load(code_ptr + low_nibble)
    val_high = tl.load(code_ptr + high_nibble)
    
    # Apply absmax scaling (block size 64)
    abs_low = tl.load(absmax_ptr + (block_start // 64), mask=mask)
    
    tl.store(out_ptr + block_start + tl.arange(0, BLOCK_SIZE // 2)*2, val_low * abs_low, mask=mask)
```

---

## Challenge B: FSDP2 + QLoRA
**Objective**: Enable Fully Sharded Data Parallel (FSDP2) training with 4-bit quantized models (QLoRA) on multi-GPU setups.

### Technical Explanation
FSDP2 shards model parameters across GPUs to save memory. In Unsloth/QLoRA:
- **Quantization**: Weights are kept in NF4, and computation is done in BFloat16.
- **Wrapping**: The `LlamaDecoderLayer` must be wrapped to ensure proper sharding boundaries.
- **CPU Offload**: Optionally offload optimizer states/parameters to CPU if VRAM is extremely limited.

### Implementation Snippet
```python
training_args = SFTConfig(
    fsdp="full_shard auto_wrap",
    fsdp_config={
        "fsdp_transformer_layer_cls_to_wrap": "LlamaDecoderLayer",
        "activation_checkpointing": True,
        "offload_params": True, # CPU Offloading
    },
    # ... other args
)
```

---

## Challenge C: torch.compile
**Objective**: Optimize the model's forward pass using `torch.compile` without graph breaks.

### Technical Explanation
`torch.compile` converts Python PyTorch code into optimized Triton kernels. 
- **Graph Breaks**: Python control flow (if/else) or unsupported ops cause "breaks" where execution falls back to slow Python.
- **Fullgraph**: Setting `fullgraph=True` ensures the entire function is compiled or errors out, which is ideal for performance.

---

## Challenge D: Llama 3.1 Tool Calling
**Objective**: Integrate official tool-calling support for Llama 3.1/3.2 into Unsloth.

### Technical Explanation
Tool calling requires specific Jinja template structures to handle `tools`, `tool_calls`, and specialized roles like `ipython`. 
- **Special Tokens**: `<|python_tag|>` (128010) and `<|eom_id|>` (128008) must be added to the tokenizer.
- **Template**: The template must format system messages with `Environment: ipython` and "Today Date" to trigger the model's capability.

### Implementation Logic
1.  **Template Update**: Replace the existing template in `unsloth/chat_templates.py` with the complex Jinja logic provided by Meta.
2.  **Special Tokens**: Patch `patch_tokenizer` in `unsloth/models/_utils.py` to check for and add the missing tokens if the model is a Llama 3 variant.

---

## Challenge E: Memory Efficient Backprop
**Objective**: Implement chunked logit computation to reduce VRAM usage by 50%.

### Technical Explanation
The standard CrossEntropy loss materializes a huge `(batch*seq, vocab)` logit tensor. For a vocab of 128k, this is massive.
- **Forward**: Compute loss in chunks of `chunk_size` (e.g., 1024 tokens). Only one chunk's logits are in memory at a time.
- **Backward**: Recompute the logits for each chunk to derive gradients for `X` and `Weight`, avoiding saving the full logit tensor.

### Implementation Snippet
```python
class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, weight, bias, labels, chunk_size):
        ctx.save_for_backward(X, weight, bias, labels)
        # Compute loss in chunks
        for i in range(0, n_tokens, chunk_size):
            logits = F.linear(X[i:i+chunk_size], weight, bias).float()
            total_loss += F.cross_entropy(logits, labels[i:i+chunk_size], reduction='sum')
        return total_loss / n_tokens

    @staticmethod
    def backward(ctx, grad_output):
        # Recompute chunk-by-chunk to save memory
        for i in range(0, n_tokens, chunk_size):
            with torch.enable_grad():
                logits = F.linear(X[i:end], weight, bias).float()
                loss = F.cross_entropy(logits, labels[i:end], reduction='sum')
                # Autograd handles dX, dWeight recomputation
```

---

##  Hardware Architecture Constraints

During verification, it was identified that the **NVIDIA P100 (Pascal)** architecture is insufficient for several advanced features in this challenge:

| Component | P100 Status | Requirement |
|-----------|-------------|-------------|
| Triton Kernels | ❌ Unsupported | SM >= 75 (T4/L4/A100) |
| BF16 Compute | ❌ Emulated | Native BF16 support |
| torch.compile | ⚠️ Degraded | Optimized for Ampere+ |
| FSDP2 | ⚠️ Slow | High-bandwidth Interconnect |

**Recommended Environments**:
- Kaggle: T4 x2 or L4.
- Colab: L4 or A100.

