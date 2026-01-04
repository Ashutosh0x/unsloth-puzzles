"""
Tests for Challenge E: Memory Efficient Backprop
"""

import pytest
import torch
import torch.nn.functional as F
import sys
from pathlib import Path

# Add challenge directory to path
challenges_dir = Path(__file__).parent.parent / "challenges" / "E_memory_backprop"
sys.path.insert(0, str(challenges_dir))

from memory_efficient import (
    MemoryEfficientLinear,
    memory_efficient_loss,
    compare_gradients
)


class TestMemoryEfficientLinear:
    """Test suite for memory-efficient cross-entropy computation."""
    
    @pytest.fixture
    def setup_tensors(self):
        """Create test tensors."""
        torch.manual_seed(42)
        batch, seq, hidden, vocab = 2, 64, 256, 1000
        hidden_states = torch.randn(batch, seq, hidden, requires_grad=True)
        lm_head = torch.nn.Linear(hidden, vocab, bias=False)
        labels = torch.randint(0, vocab, (batch, seq))
        return hidden_states, lm_head, labels
    
    def test_loss_correctness(self, setup_tensors):
        """Loss values should match standard implementation."""
        hidden_states, lm_head, labels = setup_tensors
        
        # Standard
        logits = lm_head(hidden_states)
        loss_std = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        
        # Memory efficient
        loss_eff = memory_efficient_loss(hidden_states.detach(), lm_head, labels, chunk_size=32)
        
        assert torch.allclose(loss_std, loss_eff, rtol=1e-4), \
            f"Loss mismatch: std={loss_std.item()}, eff={loss_eff.item()}"
    
    def test_gradient_correctness(self, setup_tensors):
        """Gradients should match standard implementation."""
        hidden_states, lm_head, labels = setup_tensors
        results = compare_gradients(hidden_states, lm_head, labels, chunk_size=32)
        
        assert results['loss_match'], "Loss values don't match"
        assert results['grad_match'], f"Gradients don't match, max diff: {results['max_grad_diff']}"
    
    def test_different_chunk_sizes(self, setup_tensors):
        """Results should be consistent across chunk sizes."""
        hidden_states, lm_head, labels = setup_tensors
        
        losses = []
        for chunk_size in [16, 32, 64, 128]:
            loss = memory_efficient_loss(hidden_states.detach(), lm_head, labels, chunk_size)
            losses.append(loss.item())
        
        # All should be approximately equal
        for i in range(1, len(losses)):
            assert abs(losses[i] - losses[0]) < 1e-4, \
                f"Inconsistent loss for different chunk sizes: {losses}"
    
    def test_backward_runs(self, setup_tensors):
        """Backward pass should run without errors."""
        hidden_states, lm_head, labels = setup_tensors
        hidden_states = hidden_states.detach().requires_grad_(True)
        
        loss = memory_efficient_loss(hidden_states, lm_head, labels, chunk_size=32)
        loss.backward()
        
        assert hidden_states.grad is not None, "Gradient not computed"
        assert not torch.isnan(hidden_states.grad).any(), "NaN in gradients"
        assert not torch.isinf(hidden_states.grad).any(), "Inf in gradients"


class TestVRAMReduction:
    """Tests to verify VRAM reduction (requires CUDA)."""
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_vram_usage(self):
        """Memory-efficient should use less peak VRAM."""
        torch.cuda.reset_peak_memory_stats()
        
        batch, seq, hidden, vocab = 4, 512, 1024, 32000
        
        # Standard approach
        torch.cuda.empty_cache()
        hidden_std = torch.randn(batch, seq, hidden, device='cuda', requires_grad=True)
        lm_head = torch.nn.Linear(hidden, vocab, bias=False).cuda()
        labels = torch.randint(0, vocab, (batch, seq), device='cuda')
        
        logits = lm_head(hidden_std)
        loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), labels.view(-1))
        loss.backward()
        
        peak_std = torch.cuda.max_memory_allocated()
        
        # Memory efficient approach
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        
        hidden_eff = torch.randn(batch, seq, hidden, device='cuda', requires_grad=True)
        loss_eff = memory_efficient_loss(hidden_eff, lm_head, labels, chunk_size=256)
        loss_eff.backward()
        
        peak_eff = torch.cuda.max_memory_allocated()
        
        reduction = (peak_std - peak_eff) / peak_std * 100
        print(f"VRAM reduction: {reduction:.1f}% (std={peak_std/1e6:.1f}MB, eff={peak_eff/1e6:.1f}MB)")
        
        # Should see at least 30% reduction
        assert peak_eff < peak_std * 0.7, f"Insufficient VRAM reduction: {reduction:.1f}%"


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
