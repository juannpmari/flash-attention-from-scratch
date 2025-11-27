import torch

def naive_attention(Q, K, V, is_causal=True):
    """Compute naive attention, with just torch matmul."""
    S = torch.matmul(Q, K.transpose(-2, -1)) / (Q.shape[-1] ** 0.5)
    
    if is_causal:
        seq_len = S.shape[-1]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
        S = torch.where(mask, S, torch.tensor(-1e20, device=Q.device))
        
    P = torch.softmax(S, dim=-1)
    O = torch.matmul(P, V)
    return O, P

def naive_attention_backward(Q, K, V, P, dO, is_causal=True):
    dP = torch.matmul(dO, V.transpose(-2, -1))
    dV = torch.matmul(P.transpose(-2, -1), dO)

    if is_causal:
        seq_len = Q.shape[-2]
        mask = torch.tril(torch.ones(seq_len, seq_len, device=Q.device, dtype=torch.bool))
    else:
        mask = None

    dS = P * (dP - (dP * P).sum(dim=-1, keepdim=True))
    if mask is not None:
        dS = dS * mask

    scale = (Q.shape[-1] ** -0.5)
    dQ = torch.matmul(dS, K) * scale
    dK = torch.matmul(dS.transpose(-2, -1), Q) * scale

    return dQ, dK, dV
