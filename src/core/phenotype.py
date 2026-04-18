"""
Core phenotype implementations.
This module translates Genotypes into runnable PyTorch neural modules.
"""
import torch
import torch.nn as nn
from typing import Dict, Any

class Phenotype(nn.Module):
    """The actual neural network grown from a genotype."""

    def __init__(self, genotype, input_dim: int = 64, output_dim: int = 10, module_factory=None):
        super().__init__()
        self.genotype = genotype
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.modules_dict = nn.ModuleDict()
        self.connection_graph = {}
        self.module_factory = module_factory # To prevent circular imports, use DI
        self._build_from_genotype()

    def _build_from_genotype(self):
        # Build connection graph first to determine proper inputs
        for conn in self.genotype.connections:
            if conn.target not in self.connection_graph:
                self.connection_graph[conn.target] = []
            self.connection_graph[conn.target].append(conn.source)

        module_output_dims = {}
        final_dim = self.input_dim

        for i, module_gene in enumerate(self.genotype.modules):
            # Calculate input dim based on graph
            if i == 0:
                current_dim = self.input_dim
            else:
                sources = self.connection_graph.get(module_gene.id, [])
                if sources:
                    # In forward pass, inputs are stacked and mean-pooled over dim=0 if they match length.
                    # Or minimum dimension is used. We will approximate by using the min dim of all sources.
                    source_dims = [module_output_dims.get(s, self.input_dim) for s in sources]
                    current_dim = min(source_dims)
                else:
                    # Disconnected module (should theoretically not happen if graph is valid, but fallback)
                    current_dim = self.input_dim

            if self.module_factory:
                module = self.module_factory(module_gene, current_dim)
            else:
                # Fallback simple linear
                module = nn.Sequential(nn.Linear(current_dim, 128), nn.ReLU())

            # Assume module changes dim or keeps it
            out_dim = module_gene.params.get('dim', module_gene.params.get('hidden_dim', 128))
            module_output_dims[module_gene.id] = out_dim
            self.modules_dict[module_gene.id] = module
            final_dim = out_dim

        self.output_layer = nn.Linear(final_dim, self.output_dim)

    def forward(self, x):
        activations = {}
        for i, module_gene in enumerate(self.genotype.modules):
            if i == 0:
                activations[module_gene.id] = self.modules_dict[module_gene.id](x)
            else:
                inputs = []
                for source_id in self.connection_graph.get(module_gene.id, []):
                    if source_id in activations:
                        inputs.append(activations[source_id])
                if inputs:
                    if len(inputs) == 1:
                        combined = inputs[0]
                    else:
                        min_dim = min(inp.size(-1) for inp in inputs)
                        resized = [inp[..., :min_dim] for inp in inputs]
                        combined = torch.stack(resized).mean(dim=0)
                    activations[module_gene.id] = self.modules_dict[module_gene.id](combined)

        final_module_id = self.genotype.modules[-1].id
        final_activation = activations.get(final_module_id, x)
        return self.output_layer(final_activation)
