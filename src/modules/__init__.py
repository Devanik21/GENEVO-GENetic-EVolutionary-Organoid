from .linear import LinearModule
from .attention import AttentionModule
from .rnn import RNNModule
from .cnn import CNNModule

def default_module_factory(gene, input_dim):
    if gene.type == 'linear':
        return LinearModule(gene, input_dim)
    elif gene.type == 'attention':
        return AttentionModule(gene, input_dim)
    elif gene.type == 'rnn':
        return RNNModule(gene, input_dim)
    elif gene.type == 'cnn':
        return CNNModule(gene, input_dim)
    else:
        raise ValueError(f"Unknown module type: {gene.type}")
