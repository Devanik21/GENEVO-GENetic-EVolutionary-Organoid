import torch.nn as nn

class AttentionModule(nn.Module):
    def __init__(self, gene, input_dim):
        super().__init__()
        self.gene = gene
        dim = gene.params.get('dim', 128)
        heads = gene.params.get('heads', 4)
        self.attention = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.norm = nn.LayerNorm(dim)
        self.projection = nn.Linear(input_dim, dim) if input_dim != dim else nn.Identity()
        self.recent_activity = None
        self.plasticity_active = False

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = self.projection(x)
        attn_out, _ = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        if self.plasticity_active:
            self.recent_activity = out.detach().clone()
        return out.squeeze(1) if out.size(1) == 1 else out
