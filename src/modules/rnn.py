import torch.nn as nn

class RNNModule(nn.Module):
    def __init__(self, gene, input_dim):
        super().__init__()
        self.gene = gene
        hidden_dim = gene.params.get('hidden_dim', 128)
        num_layers = gene.params.get('num_layers', 1)
        self.rnn = nn.GRU(input_dim, hidden_dim, num_layers, batch_first=True)
        self.recent_activity = None
        self.plasticity_active = False

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        out, _ = self.rnn(x)
        if self.plasticity_active:
            self.recent_activity = out.detach().clone()
        return out[:, -1, :] if out.size(1) > 1 else out.squeeze(1)
