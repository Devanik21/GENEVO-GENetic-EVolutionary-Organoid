import torch.nn as nn

class CNNModule(nn.Module):
    def __init__(self, gene, input_dim):
        super().__init__()
        self.gene = gene
        # Simple 1D conv for generality
        channels = gene.params.get('channels', 32)
        kernel_size = gene.params.get('kernel_size', 3)
        padding = kernel_size // 2

        self.conv = nn.Conv1d(input_dim, channels, kernel_size, padding=padding)
        self.activation = nn.ReLU()
        self.recent_activity = None
        self.plasticity_active = False

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        out = self.activation(self.conv(x))
        if self.plasticity_active:
            self.recent_activity = out.detach().clone()
        return out.mean(dim=-1) # pool to return flat vector
