"""
Core genotype definitions for Evolutionary Neural Systems.
This module provides the genetic representation of neural architectures.
"""
import copy
import random
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class ModuleGene:
    """Genetic specification for a neural module."""
    id: str
    type: str
    params: Dict[str, Any]

@dataclass
class ConnectionGene:
    """Genetic specification for connections between modules."""
    source: str
    target: str
    sparse: bool = False
    sparsity_level: float = 0.1
    gated: bool = False

@dataclass
class PlasticityRule:
    """Specification for local learning rules."""
    target_module: str
    rule_type: str
    learning_rate: float
    decay: float = 0.0

class Genotype:
    """Complete genetic encoding of neural architecture."""
    def __init__(self):
        self.modules: List[ModuleGene] = []
        self.connections: List[ConnectionGene] = []
        self.plasticity_rules: List[PlasticityRule] = []
        self.fitness_history: List[float] = []

    def add_module(self, module: ModuleGene):
        self.modules.append(module)

    def add_connection(self, conn: ConnectionGene):
        self.connections.append(conn)

    def add_plasticity_rule(self, rule: PlasticityRule):
        self.plasticity_rules.append(rule)

    def clone(self):
        return copy.deepcopy(self)

    def mutate(self, mutation_rate: float = 0.1):
        mutations = []
        if random.random() < mutation_rate:
            mutation_type = random.choice([
                'add_module', 'remove_module', 'modify_params',
                'add_connection', 'modify_plasticity'
            ])
            mutations.append(mutation_type)
            if mutation_type == 'add_module' and len(self.modules) < 10:
                new_id = f"M{len(self.modules)}"
                module_type = random.choice(['linear', 'attention', 'rnn', 'cnn'])
                params = self._random_params(module_type)
                self.add_module(ModuleGene(new_id, module_type, params))
            elif mutation_type == 'remove_module' and len(self.modules) > 3:
                idx = random.randint(1, len(self.modules) - 2)
                removed = self.modules.pop(idx)
                self.connections = [c for c in self.connections
                                  if c.source != removed.id and c.target != removed.id]
        return mutations

    def _random_params(self, module_type: str) -> Dict:
        if module_type == 'linear':
            return {'dim': random.choice([64, 128, 256, 512])}
        elif module_type == 'attention':
            return {'dim': random.choice([128, 256, 512]), 'heads': random.choice([2, 4, 8])}
        elif module_type == 'rnn':
            return {'hidden_dim': random.choice([64, 128, 256]), 'num_layers': random.choice([1, 2])}
        elif module_type == 'cnn':
            return {'channels': random.choice([16, 32, 64]), 'kernel_size': random.choice([3, 5])}
        return {}
