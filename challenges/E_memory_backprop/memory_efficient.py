"""
Challenge E: Memory Efficient Backprop
Goal: 50% VRAM reduction by chunked logit computation
"""

import torch
import torch.nn.functional as F
from typing import Callable, Optional


class MemoryEfficientLinear(torch.autograd.Function):
    """
    Custom autograd function for memory-efficient cross-entropy computation.
    
    Instead of materializing the full (batch, seq, vocab) logit tensor,
    we compute loss in chunks and recompute logits during backward.
    """
    
    @staticmethod
    def forward(
        ctx, 
        X: torch.Tensor,           # (batch*seq, hidden)
        weight: torch.Tensor,      # (vocab, hidden)
        bias: Optional[torch.Tensor], # (vocab,)
        labels: torch.Tensor,      # (batch*seq,)
        chunk_size: int = 1024
    ) -> torch.Tensor:
        """
        Forward pass: Compute chunked cross-entropy loss.
        """
        # Save for backward
        ctx.save_for_backward(X, weight, bias, labels)
        ctx.chunk_size = chunk_size
        
        # Backward recomputes logits by design to trade compute for memory.
        # Ensure we can actually compute gradients.
        if X.requires_grad is False or weight.requires_grad is False:
            raise RuntimeError("Hidden states and weights must require gradients for MemoryEfficientLinear.")
        
        n_tokens = X.shape[0]
        total_loss = torch.tensor(0.0, device=X.device, dtype=torch.float32)
        
        # Process in chunks to avoid OOM
        # Note: We compute the loss on the fly and don't save logits
        for i in range(0, n_tokens, chunk_size):
            end = min(i + chunk_size, n_tokens)
            chunk_x = X[i:end]
            chunk_labels = labels[i:end]
            
            # Compute logits and loss for this chunk
            logits = F.linear(chunk_x, weight, bias).float()
            chunk_loss = F.cross_entropy(logits, chunk_labels, reduction='sum')
            total_loss += chunk_loss
        
        return total_loss / n_tokens

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        X, weight, bias, labels = ctx.saved_tensors
        chunk_size = ctx.chunk_size
        
        n_tokens = X.shape[0]
        dX = torch.zeros_like(X)
        dWeight = torch.zeros_like(weight) if weight.requires_grad else None
        dBias = torch.zeros_like(bias) if bias is not None and bias.requires_grad else None
        
        for i in range(0, n_tokens, chunk_size):
            end = min(i + chunk_size, n_tokens)
            chunk_x = X[i:end].detach().requires_grad_(True)
            chunk_labels = labels[i:end]
            
            # Recompute forward for this chunk
            with torch.enable_grad():
                logits = F.linear(chunk_x, weight, bias).float()
                loss = F.cross_entropy(logits, chunk_labels, reduction='sum')
                
                # Scale the loss by (grad_output / n_tokens)
                # Since the forward pass computed the mean loss, and we want 
                # gradients of the mean loss, we use (grad_output / n_tokens).
                chunk_grad_output = grad_output / n_tokens
                
                # Using autograd.grad with a scalar output and grad_outputs=chunk_grad_output
                # handles the scaling automatically.
                tensors_to_diff = [chunk_x, weight]
                if bias is not None:
                    tensors_to_diff.append(bias)
                
                grads = torch.autograd.grad(
                    loss, 
                    tensors_to_diff,
                    grad_outputs=chunk_grad_output.expand_as(loss)
                )
            
            dX[i:end] = grads[0]
            if dWeight is not None:
                dWeight += grads[1]
            if dBias is not None:
                dBias += grads[2] if bias is not None else 0
                
        return dX, dWeight, dBias, None, None


def memory_efficient_loss(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Linear,
    labels: torch.Tensor,
    chunk_size: int = 1024
) -> torch.Tensor:
    """
    Memory-efficient language modeling loss.
    """
    # Flatten to (batch*seq, hidden)
    batch_size, seq_len, hidden_dim = hidden_states.shape
    hidden_flat = hidden_states.view(-1, hidden_dim)
    labels_flat = labels.view(-1)
    
    return MemoryEfficientLinear.apply(
        hidden_flat,
        lm_head.weight,
        lm_head.bias,
        labels_flat,
        chunk_size
    )


# Test utilities
def compare_gradients(
    hidden_states: torch.Tensor,
    lm_head: torch.nn.Linear,
    labels: torch.Tensor,
    chunk_size: int = 1024,
    rtol: float = 1e-3,
    atol: float = 1e-5
) -> dict:
    """
    Compare gradients between standard and memory-efficient implementations.
    
    Returns dict with gradient comparison metrics.
    """
    batch_size, seq_len, hidden_dim = hidden_states.shape
    labels_flat = labels.view(-1)
    
    # Clone linear layer for fair comparison
    lm_head_std = torch.nn.Linear(lm_head.in_features, lm_head.out_features, bias=lm_head.bias is not None)
    lm_head_std.load_state_dict(lm_head.state_dict())
    
    # Standard loss computation
    hidden_std = hidden_states.clone().detach().requires_grad_(True)
    logits_std = lm_head_std(hidden_std)  # (batch, seq, vocab)
    loss_std = F.cross_entropy(logits_std.view(-1, logits_std.shape[-1]), labels_flat)
    loss_std.backward()
    grad_std = hidden_std.grad.clone()
    
    # Memory efficient loss computation  
    # Need to create a fresh copy that will flow through our custom autograd
    hidden_eff = hidden_states.clone().detach().requires_grad_(True)
    hidden_flat = hidden_eff.view(-1, hidden_dim)
    
    # Use the autograd function directly
    loss_eff = MemoryEfficientLinear.apply(
        hidden_flat,
        lm_head.weight,
        lm_head.bias,
        labels_flat,
        chunk_size
    )
    loss_eff.backward()
    
    # The gradient is on hidden_eff (the 3D tensor), reshape back
    grad_eff = hidden_eff.grad.clone() if hidden_eff.grad is not None else torch.zeros_like(hidden_eff)
    
    # Compare
    loss_match = torch.allclose(loss_std, loss_eff, rtol=rtol, atol=atol)
    grad_match = torch.allclose(grad_std, grad_eff, rtol=rtol, atol=atol) if grad_eff is not None else False
    max_diff = (grad_std - grad_eff).abs().max().item() if grad_eff is not None else float('inf')
    
    return {
        'loss_standard': loss_std.item(),
        'loss_efficient': loss_eff.item(),
        'loss_match': loss_match,
        'grad_match': grad_match,
        'max_grad_diff': max_diff
    }


if __name__ == '__main__':
    # Quick test
    torch.manual_seed(42)
    
    batch, seq, hidden, vocab = 2, 128, 768, 32000
    hidden_states = torch.randn(batch, seq, hidden, requires_grad=True)
    lm_head = torch.nn.Linear(hidden, vocab, bias=False)
    labels = torch.randint(0, vocab, (batch, seq))
    
    results = compare_gradients(hidden_states, lm_head, labels, chunk_size=512)
    print("Gradient Comparison Results:")
    for k, v in results.items():
        print(f"  {k}: {v}")
