# Unsloth Puzzles - Solution Design

## Priority Order (by ROI)
1. **E: Memory Backprop** (10 pts) - Can be developed locally, algorithmic
2. **D: GitHub Issues** (12 pts) - Local development, PR-based
3. **C: torch.compile** (9 pts) - Kaggle T4, well-documented fixes
4. **A: NF4 Triton** (14 pts) - Kaggle T4, requires Triton expertise
5. **B: FSDP2 + QLoRA** (10 pts) - Requires 2×T4, most complex setup

---

## Challenge A: NF4 to Triton Kernel

### Approach
1. Study `bitsandbytes.functional.dequantize_nf4` implementation
2. Understand NF4 quantization format (4-bit, normalized float)
3. Write Triton kernel with:
   - Block-based parallelism
   - Lookup table for NF4 → float mapping
   - Fused dequant + cast to fp16/bf16

### Key Insight
```
NF4 = 4-bit index into 16-value lookup table
Each byte stores 2 NF4 values (high/low nibble)
```

### Kernel Design
```python
@triton.jit
def _your_dequantize_nf4_kernel(
    weight_ptr, out_ptr, absmax_ptr, 
    n_elements, BLOCK_SIZE: tl.constexpr
):
    # Load NF4 lookup table to shared memory
    # Parallel unpack: each thread handles BLOCK_SIZE elements
    # Dequantize: lut[index] * absmax[block_idx]
    # Write fp16/bf16 output
```

### Complexity: O(n) with high parallelism

---

## Challenge B: FSDP2 + QLoRA

### Approach
1. Use PyTorch 2.4+ FSDP2 API with `fully_shard()`
2. Wrap QLoRA adapters with FSDP2
3. Handle frozen base weights vs trainable LoRA
4. Use `MixedPrecisionPolicy` for bf16

### Key Code Pattern
```python
from torch.distributed.fsdp import fully_shard
from torch.distributed.fsdp.wrap import ModuleWrapPolicy

# Shard only LoRA modules
fsdp_model = fully_shard(
    model,
    mesh=mesh,
    mp_policy=MixedPrecisionPolicy(param_dtype=torch.bfloat16)
)
```

### Challenge: BnB quantized weights don't play well with FSDP sharding

---

## Challenge C: torch.compile without Graph Breaks

### Known Graph Break Sources
1. `bitsandbytes` CUDA ops (custom kernels)
2. Dynamic shapes in attention
3. `if` statements with tensor conditions
4. `.item()` calls
5. `torch.autograd.Function`

### Fixes
```python
# 1. Dequantize weights before compile region
# 2. Use static shapes where possible
# 3. Replace control flow with torch.where()
# 4. Wrap custom ops with fake tensor support
```

### Strategy
- Patch BnB dequantize to be compile-friendly
- Use `torch._dynamo.allow_in_graph()` for safe ops

---

## Challenge D: GitHub Issues

### Focus Areas (highest bounty first)
1. **Tool Calling** ($1000) - Implement function calling format
2. **GGUF Vision** ($500) - Add image encoder to GGUF export
3. **Attention Refactor** ($350) - Unified attention backend

### Local Development - No GPU needed

---

## Challenge E: Memory Efficient Backprop

### Core Algorithm
```
Instead of: logits = projection(hidden) -> loss(logits, labels)
Do:
  for chunk in chunks(hidden):
      chunk_logits = projection(chunk)
      partial_loss += loss(chunk_logits, chunk_labels)
  loss = partial_loss / n_chunks
```

### autograd.Function Design
```python
class MemoryEfficientLinear(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, linear, labels, forward_function, chunk_size=1024):
        # Don't save large tensors
        ctx.save_for_backward(X, labels)
        ctx.linear = linear
        ctx.chunk_size = chunk_size
        
        total_loss = 0
        for i in range(0, X.shape[0], chunk_size):
            chunk_loss = forward_function(X[i:i+chunk_size], linear, labels[i:i+chunk_size])
            total_loss += chunk_loss * (min(chunk_size, X.shape[0]-i))
        return total_loss / X.shape[0]

    @staticmethod
    def backward(ctx, grad_output):
        X, labels = ctx.saved_tensors
        linear = ctx.linear
        
        dX = torch.zeros_like(X)
        for i in range(0, X.shape[0], ctx.chunk_size):
            # Recompute forward in chunks
            chunk = X[i:i+ctx.chunk_size]
            with torch.enable_grad():
                chunk.requires_grad_(True)
                logits = linear(chunk)
                loss = F.cross_entropy(logits, labels[i:i+ctx.chunk_size])
                loss.backward()
                dX[i:i+ctx.chunk_size] = chunk.grad
        
        return dX * grad_output, None, None, None, None
```

### Key: Never materialize full vocab×seq logit tensor
