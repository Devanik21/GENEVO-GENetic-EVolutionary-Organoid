import torch.nn as nn

class LinearModule(nn.Module):
    def __init__(self, gene, input_dim):
        super().__init__()
        self.gene = gene
        dim = gene.params.get('dim', 128)
        self.layer = nn.Linear(input_dim, dim)
        self.activation = nn.ReLU()
        self.recent_activity = None
        self.plasticity_active = False

    def forward(self, x):
        out = self.activation(self.layer(x))
        if self.plasticity_active:
            self.recent_activity = out.detach().clone()
        return out
