import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def _your_dequantize_nf4_kernel(
    weight_ptr,
    absmax_ptr,
    code_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Triton kernel for NF4 dequantization.
    Unpacks 4-bit indices and scales by absmax.
    """
    pid = tl.program_id(0)
    # Each thread handles a byte (2 elements), so block handles BLOCK_SIZE elements
    # but we load BLOCK_SIZE // 2 bytes.
    block_start = pid * BLOCK_SIZE
    
    # 1. Load 4-bit weights (packed as 8-bit bytes)
    # We load BLOCK_SIZE // 2 bytes to get BLOCK_SIZE elements
    byte_offsets = (block_start // 2) + tl.arange(0, BLOCK_SIZE // 2)
    mask = byte_offsets < (n_elements // 2)
    
    packed_weights = tl.load(weight_ptr + byte_offsets, mask=mask)
    
    # 2. Unpack nibbles (low bits first for bitsandbytes)
    low_nibble = (packed_weights & 0xF).to(tl.int32)
    high_nibble = (packed_weights >> 4).to(tl.int32)
    
    # 3. Map to float using the code (LUT)
    # code_ptr is the NF4 table (16 floats)
    # We can load it once per block or per thread
    # Loading into registers for all 16 values is fast
    code_offsets = tl.arange(0, 16)
    code = tl.load(code_ptr + code_offsets)
    
    # Map nibbles to values
    # Unfortunately tl.load doesn't support indirect indexing into local registers easily
    # but we can use tl.load(code_ptr + nibble) if it's in shared memory/L1.
    val_low = tl.load(code_ptr + low_nibble)
    val_high = tl.load(code_ptr + high_nibble)
    
    # 4. Load absmax and scale
    # Bitsandbytes uses blocksize 64 by default.
    # Each absmax covers 64 elements.
    absmax_idx = (block_start + tl.arange(0, BLOCK_SIZE // 2) * 2) // 64
    # Simplification: if BLOCK_SIZE is small or aligned to 64, we can load less.
    # For now, load absmax for each element pair
    abs_low = tl.load(absmax_ptr + (block_start + tl.arange(0, BLOCK_SIZE // 2) * 2) // 64, mask=mask)
    abs_high = tl.load(absmax_ptr + (block_start + tl.arange(0, BLOCK_SIZE // 2) * 2 + 1) // 64, mask=mask)
    
    val_low = val_low * abs_low
    val_high = val_high * abs_high
    
    # 5. Store result
    out_offsets_low = block_start + tl.arange(0, BLOCK_SIZE // 2) * 2
    out_offsets_high = out_offsets_low + 1
    
    tl.store(out_ptr + out_offsets_low, val_low, mask=mask)
    tl.store(out_ptr + out_offsets_high, val_high, mask=mask)


def _your_dequantize_nf4(weight: torch.Tensor, quant_state):
    """
    Launch Triton kernel for NF4 dequantization.
    """
    n_elements = weight.numel() * 2 # weight is uint8, contains 2 nf4 values
    out = torch.empty(quant_state.shape, dtype=quant_state.dtype, device=weight.device)
    
    # Configuration
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    _your_dequantize_nf4_kernel[grid](
        weight,
        quant_state.absmax,
        quant_state.code,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return out


def your_dequantize_nf4(weight):
    """
    Entry point for the puzzle.
    """
    return _your_dequantize_nf4(weight.weight.data, weight.weight.quant_state)


# This is just a conceptual draft. I'll refine it in the implementation phase.
