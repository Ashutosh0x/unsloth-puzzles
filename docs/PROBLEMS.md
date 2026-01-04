# Unsloth Puzzles - Problem Statements

> **Source**: `Unsloth_Puzzles.ipynb` (original notebook frozen)
> **Total Max Points**: 57

---

## Challenge A: Convert `nf4` to Triton (14 pts, Hard)

### Goal
Write a **single Triton kernel** to convert `nf4` quantized tensors to `fp16`/`bf16`.

### Requirements
- Must be **≥1.15x faster** than Unsloth's `fast_dequantize` on Tesla T4
- No `torch.compile` allowed
- No large intermediate memory buffers
- Support both `fp16` and `bf16` outputs

### Skeleton
```python
@triton.jit
def _your_dequantize_nf4_kernel():
    ### TRITON CODE GOES HERE
    return

def _your_dequantize_nf4(weight, quant_state):
    ### SETUP TRITON LAUNCH HERE
    return None

def your_dequantize_nf4(weight):
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)
```

### Marking Criteria
- Kernel correctness (all dtypes)
- ≥1.15x speedup over `fast_dequantize`
- Memory efficiency

---

## Challenge B: Make QLoRA work with FSDP2 (10 pts, Medium-Hard)

### Goal
Finetune a model using **QLoRA + FSDP2** on **2x Tesla T4** GPUs.

### Requirements
- Use `transformers` and `Trainer` classes
- Loss must match single-GPU training
- Must work on Kaggle 2×T4 environment
- Demonstrate FSDP2 features: offloading, checkpointing

### Key Setup
```python
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
```

### Marking Criteria
- Working Kaggle notebook with 2×T4
- Correct FSDP2 + QLoRA integration
- Loss equivalence with single GPU

---

## Challenge C: torch.compile without graph breaks (9 pts, Easy-Medium)

### Goal
Use `torch.compile` on QLoRA finetuning with **zero graph breaks** and **<30 recompilations**.

### Requirements
- `fullgraph=True` or minimal graph breaks
- Loss curve must match non-compiled model
- Works with bitsandbytes quantization

### Starter Code
```python
torch_compile_options = {
    "epilogue_fusion": True,
    "max_autotune": True,
    "shape_padding": True,
    "trace.enabled": True,
    "triton.cudagraphs": False,
}

@torch.compile(fullgraph=True, dynamic=True, options=torch_compile_options)
def compiled_forward(model, input_ids, attention_mask, labels):
    return model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
```

### Marking Criteria
- 0 graph breaks (or minimal with justification)
- <30 total recompilations
- Correct loss values

---

## Challenge D: Help solve Unsloth issues (12 pts, Varies)

### Goal
Fix open GitHub issues in the Unsloth repository.

### Bounties Available
| Issue | Bounty | Difficulty |
|-------|--------|------------|
| Tool Calling Colab notebook | $1000 | Medium |
| GGUF Vision export | $500 | Medium |
| Refactor Attention (xformers/SDPA/flash/flex) | $350 | Medium |
| Windows support | $250 | Easy-Medium |
| Sequence Classification support | $200 | Easy |
| VLMs Data Collator optimization | $150 | Easy |
| VLMs image resizing | $100 | Easy |
| Flex Attention for dynamic sequences | $100 | Medium |

### Marking Criteria
- PR accepted to Unsloth repo
- Issue resolved completely

---

## Challenge E: Memory Efficient Backprop (10 pts, Medium-Hard)

### Goal
Reduce VRAM usage by **50%** during LLM training by computing logits on-the-fly.

### Problem
Large vocab (128k tokens) × sequence length creates huge logit tensors that OOM.

### Solution Approach
Use `torch.autograd.Function` to:
1. Compute logits in chunks during forward
2. Recompute during backward (gradient checkpointing style)
3. Never materialize full logit tensor

### Skeleton
```python
def transformation_function(batch, linear, labels):
    x = linear(batch).float()
    from torch.nn import CrossEntropyLoss
    loss_fn = CrossEntropyLoss(reduction="mean")
    loss = loss_fn(x.view(-1, x.shape[-1]), labels.view(-1))
    return loss

class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, linear, labels, forward_function):
        # EDIT THIS FUNCTION
        output = forward_function(X, linear, labels)
        ctx.save_for_backward(X)
        return output

    @staticmethod
    def backward(ctx, dY):
        X = ctx.saved_tensors
        # EDIT THIS FUNCTION
        return X, None, None, None
```

### Marking Criteria
- 50% VRAM reduction
- Correct gradients (matches non-chunked)
- Support for CE loss + other functions
- No hardcoded derivatives
