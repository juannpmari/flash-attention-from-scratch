import torch

def naive_attention(Q, K, V, is_causal=True):
    """Compute naive attention, with just torch matmul."""
    S = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    
    if is_causal:
        seq_len = S.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        S = torch.where(mask, S, torch.tensor(-1e20, device=Q.device))
        
    P = torch.softmax(S, dim=-1)
    output = torch.matmul(P, V)
    return output