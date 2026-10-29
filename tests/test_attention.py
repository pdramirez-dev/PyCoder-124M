import torch

from pycoder.model.attention import CausalSelfAttention


def test_causal_mask_and_fallback(tiny_config):
    torch.manual_seed(1)
    attn = CausalSelfAttention(tiny_config).eval()
    x = torch.randn(2, 8, tiny_config.d_model)
    modified = x.clone()
    modified[:, 4:] = torch.randn_like(modified[:, 4:])
    expected = attn(x)
    assert torch.allclose(expected[:, :4], attn(modified)[:, :4], atol=1e-6)
    attn.use_sdpa = False
    assert torch.allclose(expected, attn(x), atol=1e-5)
