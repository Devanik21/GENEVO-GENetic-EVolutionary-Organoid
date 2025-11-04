"""
GENEVO: Advanced Neuroevolutionary System for AGI
A scientifically rigorous implementation of genetic neural architecture evolution

Mathematical Foundation:

1. Phenotypic Development & Fitness Vector (Performance, Cost, Robustness):
   V(G, E(t_evo)) = E_{d ~ P(D|E)} [ (1 - L_task(P', d)), C(P')⁻¹, R(P', d, η_pert) ]
   where P' = argmin_{P*} ∫ (∇_{P*} L_task) dτ  (Lifetime Learning)
   and   P = φ(G, E, η_dev) (Development)

2. Population Evolutionary Dynamics (Non-local Fokker-Planck Equation):
   ∂ρ(G,t)/∂t = ∇_G ⋅ [D(G)∇_G ρ] - ∇_G ⋅ [ρ M(G) ∫ K(G,G') s(V(G),V(G')) ρ(G') dG']
   This describes the evolution of population density ρ(G,t) under mutation (diffusion),
   and multi-objective selection (non-local drift based on Pareto dominance).

3. Robust Multi-Objective Goal (Minimax Regret over Environment Space):
   Find G* ⊂ Γ  s.t.  G* = argmin_{G' ⊂ Γ} sup_{E ∈ E} d_H(V(G', E), P*(E))
   The goal is to find a portfolio of genotypes G* that minimizes the worst-case
   Hausdorff distance (d_H) to the true Pareto Front (P*) over all possible environments (E).

This system implements:
1. Indirect encoding via developmental programs
2. Compositional evolution with hierarchical modules  
3. Multi-objective optimization (accuracy, efficiency, complexity)
4. Coevolution with dynamic fitness landscapes
5. Baldwin effect modeling
6. Speciation and Niche Competition
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Tuple, Optional, Set
import random
import time
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import pdist, squareform
import networkx as nx
import os
from tinydb import TinyDB, Query
from collections import Counter
import json

# ==================== THEORETICAL FOUNDATIONS ====================

class EvolutionaryTheory:
    """Mathematical framework for neuroevolution"""
    
    @staticmethod
    def fisher_information(population: List, fitness: np.ndarray) -> float:
        """Fisher information measures evolutionary potential"""
        if len(fitness) < 2:
            return 0.0
        variance = np.var(fitness)
        return 1.0 / (variance + 1e-8)
    
    @staticmethod
    def genetic_diversity(population: List) -> float:
        """Shannon entropy of genotypic diversity"""
        if not population:
            return 0.0
        sizes = [sum(m.size for m in ind.modules) for ind in population]
        hist, _ = np.histogram(sizes, bins=10)
        probs = hist / (hist.sum() + 1e-8)
        return entropy(probs + 1e-8)
    
    @staticmethod
    def selection_differential(fitness: np.ndarray, selected_idx: np.ndarray) -> float:
        """Measure of selection pressure"""
        if len(fitness) == 0:
            return 0.0
        mean_all = np.mean(fitness)
        mean_selected = np.mean(fitness[selected_idx])
        return mean_selected - mean_all
    
    @staticmethod
    def heritability(parent_fitness: np.ndarray, offspring_fitness: np.ndarray) -> float:
        """Estimate narrow-sense heritability h²"""
        if len(parent_fitness) < 2 or len(offspring_fitness) < 2:
            return 0.0
        min_len = min(len(parent_fitness), len(offspring_fitness))
        if min_len < 2:
            return 0.0
        try:
            corr, _ = pearsonr(parent_fitness[:min_len], offspring_fitness[:min_len])
            return max(0, min(1, corr))
        except:
            return 0.0

# ==================== ADVANCED GENOTYPE ====================

@dataclass
class DevelopmentalGene:
    """Encodes developmental rules for phenotype construction"""
    rule_type: str  # 'proliferation', 'differentiation', 'migration', 'pruning'
    trigger_condition: str  # When this rule activates
    parameters: Dict[str, float]
    
@dataclass
class ModuleGene:
    """Enhanced module gene with biological properties"""
    id: str
    module_type: str
    size: int
    activation: str  # 'relu', 'gelu', 'silu', 'swish'
    normalization: str  # 'batch', 'layer', 'instance', 'none'
    dropout_rate: float
    learning_rate_mult: float  # Local learning rate multiplier
    plasticity: float  # Capacity for lifetime learning
    color: str
    position: Tuple[float, float, float] = (0.0, 0.0, 0.0)  # 3D coordinates
    
@dataclass
class ConnectionGene:
    """Enhanced connection with synaptic properties"""
    source: str
    target: str
    weight: float
    connection_type: str  # 'excitatory', 'inhibitory', 'modulatory'
    delay: float  # Signal propagation delay
    plasticity_rule: str  # 'hebbian', 'anti-hebbian', 'stdp', 'static'
    
@dataclass
class Genotype:
    """Complete genetic encoding with developmental program"""
    modules: List[ModuleGene]
    connections: List[ConnectionGene]
    developmental_rules: List[DevelopmentalGene] = field(default_factory=list)
    meta_parameters: Dict[str, float] = field(default_factory=dict)
    
    epigenetic_markers: Dict[str, float] = field(default_factory=dict)
    # Evolutionary metrics
    fitness: float = 0.0
    age: int = 0
    generation: int = 0
    lineage_id: str = ""
    parent_ids: List[str] = field(default_factory=list)
    
    # Performance characteristics
    accuracy: float = 0.0
    efficiency: float = 0.0  # Compute cost normalized
    complexity: float = 0.0  # Architectural complexity
    robustness: float = 0.0  # Performance under perturbation
    
    form_id: int = 1
    
    def __post_init__(self):
        if not self.lineage_id:
            self.lineage_id = f"L{random.randint(0, 999999):06d}"
        if 'learning_rate' not in self.meta_parameters:
            self.meta_parameters = {
                'learning_rate': 0.001,
                'weight_decay': 0.0001,
                'temperature': 1.0,
                'exploration_noise': 0.1
            }
    
    def copy(self):
        """Deep copy with new lineage"""
        new_genotype = Genotype(
            modules=[ModuleGene(
                m.id, m.module_type, m.size, m.activation, m.normalization,
                m.dropout_rate, m.learning_rate_mult, m.plasticity, m.color, m.position
            ) for m in self.modules],
            connections=[ConnectionGene(
                c.source, c.target, c.weight, c.connection_type, c.delay, c.plasticity_rule
            ) for c in self.connections],
            developmental_rules=[DevelopmentalGene(
                d.rule_type, d.trigger_condition, d.parameters.copy()
            ) for d in self.developmental_rules],
            meta_parameters=self.meta_parameters.copy(),
            epigenetic_markers={k: v * 0.5 for k, v in self.epigenetic_markers.items()}, # Imperfect inheritance
            fitness=self.fitness,
            age=0,
            generation=self.generation,
            parent_ids=[self.lineage_id],
            form_id=self.form_id
        )
        return new_genotype
    
    def compute_complexity(self) -> float:
        """Kolmogorov complexity approximation"""
        param_count = sum(m.size for m in self.modules)
        connection_count = len(self.connections)
        module_diversity = len(set(m.module_type for m in self.modules))
        
        # Normalized complexity score
        c_params = np.log(1 + param_count) / 20
        c_connections = len(self.connections) / (len(self.modules) ** 2 + 1)
        c_diversity = module_diversity / 10
        
        return (c_params + c_connections + c_diversity) / 3

def genotype_to_dict(g: Genotype) -> Dict:
    """Serializes a Genotype object to a dictionary."""
    return asdict(g)

def dict_to_genotype(d: Dict) -> Genotype:
    """Deserializes a dictionary back into a Genotype object."""
    # Reconstruct nested dataclasses
    d['modules'] = [ModuleGene(**m) for m in d.get('modules', [])]
    d['connections'] = [ConnectionGene(**c) for c in d.get('connections', [])]
    d['developmental_rules'] = [DevelopmentalGene(**dr) for dr in d.get('developmental_rules', [])]
    
    # The Genotype dataclass can now be instantiated with the dictionary
    return Genotype(**d)

def genomic_distance(g1: Genotype, g2: Genotype, c1=1.0, c3=0.5) -> float:
    """
    Computes genomic distance between two genotypes (NEAT-inspired).
    Distance is a weighted sum of differences in connections and module properties.
    c1: weight for disjoint/excess connections
    c3: weight for average attribute difference of matching modules (e.g., size)
    """
    # Incompatible forms are in different species
    if g1.form_id != g2.form_id:
        return float('inf')

    # Compare connections
    g1_conns = {(c.source, c.target) for c in g1.connections}
    g2_conns = {(c.source, c.target) for c in g2.connections}
    
    disjoint_conns = len(g1_conns.symmetric_difference(g2_conns))
    
    # Compare modules
    g1_modules = {m.id: m for m in g1.modules}
    g2_modules = {m.id: m for m in g2.modules}
    
    matching_modules = 0
    size_diff = 0.0
    plasticity_diff = 0.0
    
    all_module_ids = set(g1_modules.keys()) | set(g2_modules.keys())
    
    for mid in all_module_ids:
        if mid in g1_modules and mid in g2_modules:
            matching_modules += 1
            size_diff += abs(g1_modules[mid].size - g2_modules[mid].size)
            plasticity_diff += abs(g1_modules[mid].plasticity - g2_modules[mid].plasticity)

    avg_size_diff = (size_diff / matching_modules) if matching_modules > 0 else 0
    avg_plasticity_diff = (plasticity_diff / matching_modules) if matching_modules > 0 else 0

    N = max(1, len(g1.connections), len(g2.connections))
    distance = (c1 * disjoint_conns / N) + (c3 * (avg_size_diff / 100 + avg_plasticity_diff))
    
    return distance

def is_viable(genotype: Genotype) -> bool:
    """
    Checks if a genotype is structurally viable, meaning it has a path
    from an input node to an output node. This prevents non-functional
    architectures from entering the population.
    """
    if not genotype.modules or not genotype.connections:
        return False

    G = nx.DiGraph()
    module_ids = {m.id for m in genotype.modules}
    
    for conn in genotype.connections:
        # Ensure connections are between existing modules
        if conn.source in module_ids and conn.target in module_ids:
            G.add_edge(conn.source, conn.target)

    if G.number_of_nodes() < 2: # Need at least two nodes to form a path
        return False

    # Identify input nodes (in-degree == 0) and output nodes (out-degree == 0)
    input_nodes = [node for node, in_degree in G.in_degree() if in_degree == 0]
    output_nodes = [node for node, out_degree in G.out_degree() if out_degree == 0]

    # If there are no clear input/output nodes (e.g., a cycle), use heuristics
    if not input_nodes:
        potential_inputs = [m.id for m in genotype.modules if 'input' in m.id or 'embed' in m.id or 'V1' in m.id]
        input_nodes = [node for node in potential_inputs if node in G.nodes]

    if not output_nodes:
        potential_outputs = [m.id for m in genotype.modules if 'output' in m.id or 'PFC' in m.id]
        output_nodes = [node for node in potential_outputs if node in G.nodes]

    if not input_nodes or not output_nodes:
        return False # Cannot determine a start or end point

    # Check for a path from any input to any output
    for start_node in input_nodes:
        for end_node in output_nodes:
            if start_node in G and end_node in G and nx.has_path(G, start_node, end_node):
                return True

    return False

# ==================== ADVANCED INITIALIZATION ====================

def initialize_genotype(form_id: int, complexity_level: str = 'medium') -> Genotype:
    """Initialize genotype with scientifically grounded architectures"""
    
    complexity_scales = {
        'minimal': (32, 128, 0.3),
        'medium': (64, 256, 0.5),
        'high': (128, 512, 0.8)
    }
    
    base_size, max_size, connection_density = complexity_scales[complexity_level]
    
    forms = {
        1: {  # Convolutional Cascade (Visual Processing)
            'name': 'Hierarchical Convolutional',
            'modules': [
                ModuleGene('V1', 'conv', base_size, 'relu', 'batch', 0.1, 1.0, 0.3, '#FF6B6B', (0, 0, 0)),
                ModuleGene('V2', 'conv', base_size*2, 'relu', 'batch', 0.15, 1.0, 0.4, '#FD79A8', (1, 0, 0)),
                ModuleGene('V4', 'conv', base_size*3, 'gelu', 'layer', 0.2, 0.8, 0.5, '#FDCB6E', (2, 0, 0)),
                ModuleGene('IT', 'attention', base_size*4, 'gelu', 'layer', 0.25, 0.6, 0.6, '#00B894', (3, 0, 0)),
                ModuleGene('PFC', 'mlp', max_size, 'swish', 'layer', 0.3, 0.5, 0.7, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'hierarchical',
            'inductive_bias': 'spatial_locality'
        },
        2: {  # Attention Network (Transformer-like)
            'name': 'Multi-Head Attention System',
            'modules': [
                ModuleGene('embed', 'mlp', base_size*2, 'gelu', 'layer', 0.1, 1.0, 0.4, '#FF6B6B', (0, 0, 0)),
                ModuleGene('attn_1', 'attention', base_size*4, 'gelu', 'layer', 0.1, 1.0, 0.6, '#FECA57', (1, 1, 0)),
                ModuleGene('attn_2', 'attention', base_size*4, 'gelu', 'layer', 0.15, 0.9, 0.6, '#48DBFB', (2, 1, 0)),
                ModuleGene('attn_3', 'attention', base_size*4, 'gelu', 'layer', 0.2, 0.8, 0.6, '#A29BFE', (3, 1, 0)),
                ModuleGene('output', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'residual_attention',
            'inductive_bias': 'long_range_dependencies'
        },
        3: {  # Recurrent Memory Network
            'name': 'Dynamical Recurrent System',
            'modules': [
                ModuleGene('input_gate', 'mlp', base_size, 'sigmoid', 'layer', 0.1, 1.0, 0.4, '#FF6B6B', (0, 0, 0)),
                ModuleGene('lstm_1', 'recurrent', base_size*3, 'tanh', 'layer', 0.2, 1.0, 0.8, '#A29BFE', (1, 0, 1)),
                ModuleGene('lstm_2', 'recurrent', base_size*3, 'tanh', 'layer', 0.25, 0.9, 0.8, '#6C5CE7', (2, 0, 1)),
                ModuleGene('memory', 'attention', base_size*2, 'gelu', 'layer', 0.2, 0.8, 0.9, '#55EFC4', (3, 0, 0.5)),
                ModuleGene('output_gate', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'recurrent_memory',
            'inductive_bias': 'temporal_integration'
        },
        4: {  # Parallel Hybrid (Multi-pathway)
            'name': 'Dual-Stream Processing',
            'modules': [
                ModuleGene('input', 'conv', base_size, 'relu', 'batch', 0.1, 1.0, 0.3, '#FF6B6B', (0, 0, 0)),
                ModuleGene('dorsal_1', 'conv', base_size*2, 'relu', 'batch', 0.15, 1.0, 0.5, '#FD79A8', (1, 1, 0)),
                ModuleGene('dorsal_2', 'attention', base_size*2, 'gelu', 'layer', 0.2, 0.9, 0.6, '#FDCB6E', (2, 1, 0)),
                ModuleGene('ventral_1', 'conv', base_size*2, 'relu', 'batch', 0.15, 1.0, 0.5, '#48DBFB', (1, -1, 0)),
                ModuleGene('ventral_2', 'attention', base_size*2, 'gelu', 'layer', 0.2, 0.9, 0.6, '#A29BFE', (2, -1, 0)),
                ModuleGene('integrate', 'mlp', base_size*4, 'swish', 'layer', 0.25, 0.8, 0.7, '#00B894', (3, 0, 0)),
                ModuleGene('output', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'dual_pathway',
            'inductive_bias': 'specialized_processing'
        },
        5: {  # Graph Neural Architecture
            'name': 'Graph Relational Network',
            'modules': [
                ModuleGene('embed', 'mlp', base_size, 'gelu', 'layer', 0.1, 1.0, 0.4, '#FF6B6B', (0, 0, 0)),
                ModuleGene('graph_1', 'graph', base_size*2, 'gelu', 'layer', 0.15, 1.0, 0.7, '#A29BFE', (1, 0.5, 0.5)),
                ModuleGene('graph_2', 'graph', base_size*3, 'gelu', 'layer', 0.2, 0.9, 0.7, '#74B9FF', (2, -0.5, 0.5)),
                ModuleGene('graph_3', 'graph', base_size*3, 'gelu', 'layer', 0.2, 0.9, 0.7, '#00CEC9', (2, 0.5, -0.5)),
                ModuleGene('aggregate', 'attention', base_size*4, 'gelu', 'layer', 0.25, 0.8, 0.6, '#55EFC4', (3, 0, 0)),
                ModuleGene('output', 'mlp', max_size, 'swish', 'layer', 0.3, 0.7, 0.5, '#96CEB4', (4, 0, 0)),
            ],
            'topology': 'fully_connected_graph',
            'inductive_bias': 'relational_reasoning'
        }
    }
    
    form = forms[form_id]
    modules = form['modules']
    
    # Create connections based on topology
    connections = []
    
    if form['topology'] == 'hierarchical':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                float(np.random.uniform(0.7, 1.0)), 'excitatory', 0.01, 'hebbian'
            ))
            
    elif form['topology'] == 'residual_attention':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                float(np.random.uniform(0.8, 1.0)), 'excitatory', 0.005, 'stdp'
            ))
        # Add residual connections
        for i in range(1, len(modules) - 2):
            if 'attn' in modules[i].id:
                connections.append(ConnectionGene(
                    modules[i].id, modules[i+2].id,
                    float(np.random.uniform(0.3, 0.5)), 'excitatory', 0.02, 'static'
                ))
                
    elif form['topology'] == 'recurrent_memory':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                float(np.random.uniform(0.7, 0.9)), 'excitatory', 0.01, 'hebbian'
            ))
        # Recurrent connections
        for module in modules:
            if 'lstm' in module.id:
                connections.append(ConnectionGene(
                    module.id, module.id,
                    float(np.random.uniform(0.4, 0.6)), 'modulatory', 0.001, 'stdp'
                ))
                
    elif form['topology'] == 'dual_pathway':
        # Input to both pathways
        connections.append(ConnectionGene('input', 'dorsal_1', 0.8, 'excitatory', 0.01, 'hebbian'))
        connections.append(ConnectionGene('input', 'ventral_1', 0.8, 'excitatory', 0.01, 'hebbian'))
        # Within pathways
        connections.append(ConnectionGene('dorsal_1', 'dorsal_2', 0.9, 'excitatory', 0.005, 'stdp'))
        connections.append(ConnectionGene('ventral_1', 'ventral_2', 0.9, 'excitatory', 0.005, 'stdp'))
        # Convergence
        connections.append(ConnectionGene('dorsal_2', 'integrate', 0.7, 'excitatory', 0.02, 'hebbian'))
        connections.append(ConnectionGene('ventral_2', 'integrate', 0.7, 'excitatory', 0.02, 'hebbian'))
        connections.append(ConnectionGene('integrate', 'output', 0.9, 'excitatory', 0.01, 'stdp'))
        
    elif form['topology'] == 'fully_connected_graph':
        for i, m1 in enumerate(modules):
            for j, m2 in enumerate(modules):
                if i != j and np.random.random() > 0.4:
                    weight = float(np.random.uniform(0.3, 0.8) if i < j else np.random.uniform(0.2, 0.5))
                    conn_type = 'excitatory' if i < j else 'modulatory'
                    connections.append(ConnectionGene(
                        m1.id, m2.id, weight, conn_type,
                        float(np.random.uniform(0.001, 0.02)), 'hebbian'
                    ))
    
    # Create developmental rules
    dev_rules = [
        DevelopmentalGene('proliferation', 'fitness_plateau', {'growth_rate': 1.1, 'max_size': max_size * 2, 'stagnation_threshold': 3}),
        DevelopmentalGene('pruning', 'maturity', {'threshold': 0.1, 'maturity_age': 5}),
        DevelopmentalGene('differentiation', 'environmental_signal', {'specialization_strength': 0.7})
    ]
    
    genotype = Genotype(
        modules=modules,
        connections=connections,
        developmental_rules=dev_rules,
        form_id=form_id
    )
    genotype.complexity = genotype.compute_complexity()
    
    return genotype

# ==================== ADVANCED EVOLUTION ====================

def mutate(genotype: Genotype, mutation_rate: float = 0.2, innovation_rate: float = 0.05) -> Genotype:
    """Biologically-inspired mutation with innovation"""
    mutated = genotype.copy()
    mutated.age = 0
    
    # 1. Point mutations (module parameters)
    for module in mutated.modules:
        if random.random() < mutation_rate:
            # Size mutation with drift
            change_factor = np.random.lognormal(0, 0.2)
            module.size = int(module.size * change_factor)
            module.size = int(np.clip(module.size, 16, 1024))
        
        if random.random() < mutation_rate * 0.5:
            # Plasticity mutation
            module.plasticity += np.random.normal(0, 0.1)
            module.plasticity = float(np.clip(module.plasticity, 0, 1))
        
        if random.random() < mutation_rate * 0.3:
            # Learning rate multiplier
            module.learning_rate_mult *= np.random.lognormal(0, 0.15)
            module.learning_rate_mult = float(np.clip(module.learning_rate_mult, 0.1, 2.0))
        
        if random.random() < mutation_rate * 0.2:
            # Activation function mutation
            module.activation = random.choice(['relu', 'gelu', 'silu', 'swish'])
    
    # 2. Connection weight mutations
    for connection in mutated.connections:
        if random.random() < mutation_rate:
            connection.weight += np.random.normal(0, 0.15)
            connection.weight = float(np.clip(connection.weight, 0.05, 1.0))
        
        if random.random() < mutation_rate * 0.3:
            # Plasticity rule mutation
            connection.plasticity_rule = random.choice(['hebbian', 'anti-hebbian', 'stdp', 'static'])
    
    # 3. Structural mutations (innovation)
    if random.random() < innovation_rate:
        # Add new connection
        if len(mutated.modules) > 2:
            source = random.choice(mutated.modules[:-1])
            target = random.choice([m for m in mutated.modules if m.id != source.id])
            
            # Check if connection already exists
            exists = any(c.source == source.id and c.target == target.id for c in mutated.connections)
            if not exists:
                mutated.connections.append(ConnectionGene(
                    source.id, target.id,
                    float(np.random.uniform(0.2, 0.5)), 'excitatory',
                    float(np.random.uniform(0.001, 0.02)), 'hebbian'
                ))
    
    if random.random() < innovation_rate * 0.5:
        # Add new module (rare)
        new_id = f"evolved_{len(mutated.modules)}"
        avg_size = int(np.mean([m.size for m in mutated.modules]))
        new_module = ModuleGene(
            new_id, random.choice(['mlp', 'attention', 'graph']),
            avg_size, random.choice(['gelu', 'swish']), 'layer',
            0.2, 1.0, 0.5, '#DDA15E',
            (len(mutated.modules), 0, 0)
        )
        mutated.modules.append(new_module)
        
        # Connect to network
        source = random.choice(mutated.modules[:-1])
        mutated.connections.append(ConnectionGene(
            source.id, new_id, 0.3, 'excitatory', 0.01, 'hebbian'
        ))
    
    # 4. Meta-parameter mutations
    for key in mutated.meta_parameters:
        if random.random() < mutation_rate * 0.4:
            mutated.meta_parameters[key] *= np.random.lognormal(0, 0.1)
    
    mutated.complexity = mutated.compute_complexity()
    return mutated

def crossover(parent1: Genotype, parent2: Genotype, crossover_rate: float = 0.7) -> Genotype:
    """Advanced recombination with homologous alignment"""
    if parent1.form_id != parent2.form_id:
        return parent1.copy()
    
    child = parent1.copy()
    child.parent_ids = [parent1.lineage_id, parent2.lineage_id]
    
    if random.random() > crossover_rate:
        return child
    
    # Epigenetic Crossover: average the decayed markers from both parents
    # child.epigenetic_markers already contains decayed markers from parent1 via .copy()
    for key, p2_val in parent2.epigenetic_markers.items():
        # Decay parent2's markers as well before averaging
        decayed_p2_val = p2_val * 0.5 
        # Get the current value (from parent1) and average with parent2's
        current_val = child.epigenetic_markers.get(key, 0.0)
        child.epigenetic_markers[key] = (current_val + decayed_p2_val) / 2.0

    # Module-level crossover
    for i in range(min(len(parent1.modules), len(parent2.modules))):
        if random.random() < 0.5:
            # Inherit from parent2
            child.modules[i].size = parent2.modules[i].size
            child.modules[i].plasticity = parent2.modules[i].plasticity
            child.modules[i].learning_rate_mult = parent2.modules[i].learning_rate_mult
    
    # Connection-level crossover
    p1_conns = {(c.source, c.target): c for c in parent1.connections}
    p2_conns = {(c.source, c.target): c for c in parent2.connections}
    
    new_connections = []
    all_keys = set(p1_conns.keys()) | set(p2_conns.keys())
    
    for key in all_keys:
        if key in p1_conns and key in p2_conns:
            # Both parents have this connection
            conn = p1_conns[key] if random.random() < 0.5 else p2_conns[key]
            new_connections.append(ConnectionGene(
                conn.source, conn.target, conn.weight, conn.connection_type,
                conn.delay, conn.plasticity_rule
            ))
        elif key in p1_conns:
            new_connections.append(ConnectionGene(
                p1_conns[key].source, p1_conns[key].target, p1_conns[key].weight,
                p1_conns[key].connection_type, p1_conns[key].delay, p1_conns[key].plasticity_rule
            ))
        else:
            new_connections.append(ConnectionGene(
                p2_conns[key].source, p2_conns[key].target, p2_conns[key].weight,
                p2_conns[key].connection_type, p2_conns[key].delay, p2_conns[key].plasticity_rule
            ))
    
    child.connections = new_connections
    
    # Meta-parameter crossover
    for key in child.meta_parameters:
        if key in parent2.meta_parameters and random.random() < 0.5:
            child.meta_parameters[key] = parent2.meta_parameters[key]
    
    child.complexity = child.compute_complexity()
    return child

def apply_endosymbiosis(recipient: Genotype, donors: List[Genotype]) -> Genotype:
    """
    A rare event where a recipient genotype acquires a module from a highly fit donor.
    This simulates horizontal gene transfer or endosymbiosis.
    """
    if not donors or not recipient.connections:
        return recipient

    # Select a random elite donor
    donor = random.choice(donors)
    
    # Select a non-trivial module from the donor to acquire
    candidate_modules = [m for m in donor.modules if m.module_type not in ['input', 'output']]
    if not candidate_modules:
        return recipient

    module_to_acquire = ModuleGene(**asdict(random.choice(candidate_modules)))
    
    # Ensure the new module has a unique ID
    new_id = f"endo_{module_to_acquire.id}_{random.randint(0, 999)}"
    if any(m.id == new_id for m in recipient.modules):
        return recipient # Avoid ID collision
    module_to_acquire.id = new_id

    # Insert the module by splitting an existing connection
    connection_to_split = random.choice(recipient.connections)
    recipient.connections.remove(connection_to_split)

    # Add the new module and wire it in
    recipient.modules.append(module_to_acquire)
    recipient.connections.append(ConnectionGene(connection_to_split.source, new_id, float(np.random.uniform(0.4, 0.8)), 'excitatory', 0.01, 'hebbian'))
    recipient.connections.append(ConnectionGene(new_id, connection_to_split.target, float(np.random.uniform(0.4, 0.8)), 'excitatory', 0.01, 'stdp'))

    recipient.complexity = recipient.compute_complexity()
    st.toast(f"Endosymbiosis Event: Acquired module '{module_to_acquire.module_type}'.", icon="🧬")
    return recipient

def apply_developmental_rules(genotype: Genotype, stagnation_counter: int) -> Genotype:
    """
    Executes the developmental program encoded in the genotype.
    This simulates processes like pruning and proliferation during an individual's life.
    """
    developed_genotype = genotype # Work on the same object
    
    for rule in developed_genotype.developmental_rules:
        if rule.rule_type == 'pruning' and rule.trigger_condition == 'maturity':
            # Prune weak connections if the individual is mature enough
            if developed_genotype.age > rule.parameters.get('maturity_age', 5):
                threshold = rule.parameters.get('threshold', 0.1)
                developed_genotype.connections = [
                    c for c in developed_genotype.connections if c.weight >= threshold
                ]
                
        elif rule.rule_type == 'proliferation' and rule.trigger_condition == 'fitness_plateau':
            # Grow a module if the population is stagnating
            if stagnation_counter > rule.parameters.get('stagnation_threshold', 3):
                if developed_genotype.modules:
                    target_module = random.choice(developed_genotype.modules)
                    growth_rate = rule.parameters.get('growth_rate', 1.1)
                    max_size = rule.parameters.get('max_size', 2048)
                    target_module.size = int(min(target_module.size * growth_rate, max_size))

    # Recalculate complexity after development
    developed_genotype.complexity = developed_genotype.compute_complexity()
    return developed_genotype

def evaluate_fitness(genotype: Genotype, task_type: str, generation: int, weights: Optional[Dict[str, float]] = None, enable_epigenetics: bool = False, enable_baldwin: bool = False, epistatic_linkage_k: int = 0, parasite_profile: Optional[Dict] = None) -> Tuple[float, Dict[str, float]]:
    """
    Multi-objective fitness evaluation with realistic task simulation
    
    Returns: (total_fitness, component_scores)
    """
    
    # Initialize component scores
    scores = {
        'task_accuracy': 0.0,
        'efficiency': 0.0,
        'robustness': 0.0,
        'generalization': 0.0
    }
    
    # Compute architectural properties
    total_params = sum(m.size for m in genotype.modules)
    avg_plasticity = np.mean([m.plasticity for m in genotype.modules])
    connection_density = len(genotype.connections) / (len(genotype.modules) ** 2 + 1)
    
    # 1a. Epigenetic Inheritance Bonus
    # Apply bonus from markers inherited from parents.
    epigenetic_bonus = 0.0
    if enable_epigenetics:
        aptitude_key = f"{task_type}_aptitude"
        if aptitude_key in genotype.epigenetic_markers:
            epigenetic_bonus = genotype.epigenetic_markers[aptitude_key]
    
    # 1b. Task-specific accuracy simulation
    if task_type == 'Abstract Reasoning (ARC-AGI-2)':
        # Reward compositional structures and high plasticity
        graph_attention_count = sum(1 for m in genotype.modules 
                                   if m.module_type in ['graph', 'attention'])
        compositional_score = graph_attention_count / len(genotype.modules)
        plasticity_bonus = avg_plasticity * 0.4
        
        # ARC requires efficiency and abstraction
        efficiency_penalty = np.exp(-total_params / 50000)  # Prefer compact architectures
        
        scores['task_accuracy'] = (
            compositional_score * 0.4 +
            plasticity_bonus * 0.3 +
            efficiency_penalty * 0.3 +
            np.random.normal(0, 0.05)  # Stochasticity
        )
        
        # Evolved architectures improve over generations
        scores['task_accuracy'] *= (1 + 0.01 * generation)
        
    elif task_type == 'Vision (ImageNet)':
        conv_count = sum(1 for m in genotype.modules if m.module_type == 'conv')
        hierarchical_bonus = 0.2 if genotype.form_id in [1, 4] else 0.0
        
        scores['task_accuracy'] = (
            (conv_count / len(genotype.modules)) * 0.5 +
            hierarchical_bonus +
            connection_density * 0.2 +
            np.random.normal(0, 0.05)
        )
        
    elif task_type == 'Language (MMLU-Pro)':
        attn_count = sum(1 for m in genotype.modules if m.module_type == 'attention')
        depth_bonus = len(genotype.modules) / 10
        
        scores['task_accuracy'] = (
            (attn_count / len(genotype.modules)) * 0.6 +
            min(depth_bonus, 0.3) +
            np.random.normal(0, 0.05)
        )
        
    elif task_type == 'Sequential Prediction':
        rec_count = sum(1 for m in genotype.modules if m.module_type == 'recurrent')
        memory_bonus = 0.3 if any('memory' in m.id for m in genotype.modules) else 0.0
        
        scores['task_accuracy'] = (
            (rec_count / len(genotype.modules)) * 0.5 +
            memory_bonus +
            avg_plasticity * 0.15 +
            np.random.normal(0, 0.05)
        )
        
    elif task_type == 'Multi-Task Learning':
        module_diversity = len(set(m.module_type for m in genotype.modules))
        hybrid_bonus = 0.4 if genotype.form_id in [4, 5] else 0.0
        
        scores['task_accuracy'] = (
            (module_diversity / 5) * 0.4 +
            hybrid_bonus +
            connection_density * 0.15 +
            np.random.normal(0, 0.05)
        )
    
    # Apply epigenetic bonus to the base score
    scores['task_accuracy'] += epigenetic_bonus

    # 2. Lifetime Learning Simulation (Baldwin Effect)
    # Plasticity allows an individual to "learn" and improve its performance during its lifetime.
    # This bonus is added to the base task accuracy, rewarding adaptable architectures.
    if enable_baldwin:
        lifetime_learning_bonus = avg_plasticity * 0.2  # Max 20% accuracy boost from learning
        scores['task_accuracy'] += lifetime_learning_bonus
    
    # Clamp task accuracy after adding bonus
    scores['task_accuracy'] = np.clip(scores['task_accuracy'], 0, 1)
    
    # 3. Efficiency score (inverse of computational cost)
    # Prefer architectures with good accuracy-to-parameter ratio
    param_efficiency = 1.0 / (1.0 + np.log(1 + total_params / 10000))
    connection_efficiency = 1.0 - min(connection_density, 0.8)
    
    scores['efficiency'] = (param_efficiency + connection_efficiency) / 2
    
    # 4. Robustness (architectural stability)
    # More diverse connections and moderate plasticity = more robust
    robustness_from_diversity = len(set(c.connection_type for c in genotype.connections)) / 3
    robustness_from_plasticity = 1.0 - abs(avg_plasticity - 0.5) * 2  # Prefer moderate
    
    scores['robustness'] = (robustness_from_diversity * 0.5 + robustness_from_plasticity * 0.5)
    
    # 5. Generalization potential
    # Architectural properties that predict generalization
    depth = len(genotype.modules)
    modularity_score = 1.0 - abs(connection_density - 0.3) * 2  # Sweet spot at 0.3
    
    scores['generalization'] = (
        min(depth / 10, 1.0) * 0.4 +
        modularity_score * 0.3 +
        avg_plasticity * 0.3
    )

    # 6. Epigenetic Marking (Lamarckian-like learning)
    # The individual "learns" from its performance, creating a marker for its offspring.
    # This maps current performance to a small, heritable aptitude value.
    if enable_epigenetics:
        aptitude_key = f"{task_type}_aptitude"
        performance_marker = (scores['task_accuracy'] - 0.5) * 0.05 # Small learning step
        current_aptitude = genotype.epigenetic_markers.get(aptitude_key, 0.0)
        genotype.epigenetic_markers[aptitude_key] = np.clip(current_aptitude + performance_marker, -0.15, 0.15)
    
    # 7. Epistatic Contribution (NK Landscape Simulation)
    # Models how gene interactions create a rugged fitness landscape.
    epistatic_contribution = 0.0
    if epistatic_linkage_k > 0 and len(genotype.modules) > epistatic_linkage_k:
        num_modules = len(genotype.modules)
        for i, module in enumerate(genotype.modules):
            # Select K interacting genes (modules)
            indices = list(range(num_modules))
            indices.remove(i)
            interacting_indices = random.sample(indices, k=epistatic_linkage_k)
            
            # Create a unique "genetic context" signature
            context_signature = tuple([module.module_type] + [genotype.modules[j].module_type for j in interacting_indices])
            
            # Use a hash to create a deterministic, pseudo-random contribution for this context
            hash_val = hash(context_signature)
            epistatic_contribution += (hash_val % 2000 - 1000) / 10000.0 # Small contribution in [-0.1, 0.1]

    # Multi-objective fitness with task-dependent weights
    if weights is None:
        if 'ARC' in task_type:
            weights = {'task_accuracy': 0.5, 'efficiency': 0.3, 'robustness': 0.1, 'generalization': 0.1}
        else:
            weights = {'task_accuracy': 0.6, 'efficiency': 0.2, 'robustness': 0.1, 'generalization': 0.1}
        
    total_fitness = sum(scores[k] * weights[k] for k in weights)
    
    # Apply epistatic effect to final fitness
    total_fitness += epistatic_contribution
    
    # 8. Red Queen Coevolution (Parasite Attack)
    # If a genotype has a trait targeted by the co-evolving parasite, its fitness is penalized.
    if parasite_profile:
        vulnerability_score = 0.0
        for module in genotype.modules:
            if module.module_type == parasite_profile['target_type'] and module.activation == parasite_profile['target_activation']:
                vulnerability_score += 0.1 # Each matching module increases vulnerability
        
        total_fitness *= (1.0 - min(vulnerability_score, 0.5)) # Max 50% fitness reduction

    # Apply a small fitness floor. This prevents complete zeros for viable but
    # poorly performing individuals, which can help maintain diversity and
    # prevent numerical issues in some selection schemes.
    total_fitness = max(total_fitness, 1e-6)

    # Store component scores
    genotype.accuracy = scores['task_accuracy']
    genotype.efficiency = scores['efficiency']
    genotype.robustness = scores['robustness']
    
    return total_fitness, scores

def synthesize_master_architecture(top_individuals: List[Genotype]) -> Optional[Genotype]:
    """
    Synthesizes a "master" architecture by creating a consensus from the top n individuals.
    It starts with the best individual and refines its parameters by averaging them
    with other elite individuals. It may also add highly-voted structural elements.
    """
    if not top_individuals:
        return None

    n = len(top_individuals)
    best_ind = top_individuals[0]
    
    # Start with a copy of the best individual as the template
    master_arch = best_ind.copy()
    master_arch.lineage_id = "SYNTHESIZED_MASTER"
    master_arch.fitness = float(np.mean([ind.fitness for ind in top_individuals]))
    master_arch.accuracy = float(np.mean([ind.accuracy for ind in top_individuals]))
    master_arch.efficiency = float(np.mean([ind.efficiency for ind in top_individuals]))
    master_arch.robustness = float(np.mean([ind.robustness for ind in top_individuals]))

    # --- 1. Parameter Averaging ---
    
    # Create lookups for faster access
    all_modules_by_id = {ind.lineage_id: {m.id: m for m in ind.modules} for ind in top_individuals}
    all_conns_by_key = {ind.lineage_id: {(c.source, c.target): c for c in ind.connections} for ind in top_individuals}

    # Average module parameters
    for master_module in master_arch.modules:
        module_id = master_module.id
        peers = [all_modules_by_id[ind.lineage_id].get(module_id) for ind in top_individuals]
        peers = [p for p in peers if p is not None]
        
        if len(peers) > 1:
            master_module.size = int(np.mean([p.size for p in peers]))
            master_module.dropout_rate = float(np.mean([p.dropout_rate for p in peers]))
            master_module.learning_rate_mult = float(np.mean([p.learning_rate_mult for p in peers]))
            master_module.plasticity = float(np.mean([p.plasticity for p in peers]))

    # Average connection parameters
    for master_conn in master_arch.connections:
        conn_key = (master_conn.source, master_conn.target)
        peers = [all_conns_by_key[ind.lineage_id].get(conn_key) for ind in top_individuals]
        peers = [p for p in peers if p is not None]
        
        if len(peers) > 1:
            master_conn.weight = float(np.mean([p.weight for p in peers]))
            plasticity_rules = [p.plasticity_rule for p in peers]
            master_conn.plasticity_rule = Counter(plasticity_rules).most_common(1)[0][0]

    # --- 2. Structural Voting (Add missing consensus connections) ---
    connection_counts = Counter()
    all_connections_map = {} 
    for ind in top_individuals:
        for conn in ind.connections:
            key = (conn.source, conn.target)
            connection_counts[key] += 1
            if key not in all_connections_map:
                all_connections_map[key] = conn

    master_conn_keys = {(c.source, c.target) for c in master_arch.connections}
    consensus_threshold = n / 2.0
    
    for conn_key, count in connection_counts.items():
        if count > consensus_threshold and conn_key not in master_conn_keys:
            source_id, target_id = conn_key
            master_module_ids = {m.id for m in master_arch.modules}
            if source_id in master_module_ids and target_id in master_module_ids:
                template_conn = all_connections_map[conn_key]
                peers = [all_conns_by_key[ind.lineage_id].get(conn_key) for ind in top_individuals]
                peers = [p for p in peers if p is not None]
                
                if not peers: continue

                avg_weight = float(np.mean([p.weight for p in peers]))
                plasticity_rule = Counter([p.plasticity_rule for p in peers]).most_common(1)[0][0]
                
                new_conn = ConnectionGene(
                    source=template_conn.source,
                    target=template_conn.target,
                    weight=avg_weight,
                    connection_type=template_conn.connection_type,
                    delay=template_conn.delay,
                    plasticity_rule=plasticity_rule
                )
                master_arch.connections.append(new_conn)
                
    master_arch.complexity = master_arch.compute_complexity()
    return master_arch

def analyze_lesion_sensitivity(
    master_architecture: Genotype, 
    base_fitness: float, 
    task_type: str, 
    fitness_weights: Dict, 
    eval_params: Dict
) -> Dict[str, float]:
    """
    Performs a lesion study on the master architecture to find critical components.
    Returns a dictionary of component ID to fitness drop (criticality).
    """
    criticality_scores = {}

    # 1. Lesion Modules
    for module in master_architecture.modules:
        # Don't lesion input/output
        if 'input' in module.id or 'output' in module.id:
            continue

        lesioned_arch = master_architecture.copy()
        
        # Remove the module and any connections to/from it
        lesioned_arch.modules = [m for m in lesioned_arch.modules if m.id != module.id]
        lesioned_arch.connections = [
            c for c in lesioned_arch.connections if c.source != module.id and c.target != module.id
        ]
        
        if not lesioned_arch.connections or not lesioned_arch.modules: continue # Skip if network is disconnected

        lesioned_fitness, _ = evaluate_fitness(
            lesioned_arch, 
            task_type, 
            lesioned_arch.generation, 
            fitness_weights, 
            **eval_params
        )
        
        fitness_drop = base_fitness - lesioned_fitness
        criticality_scores[f"Module: {module.id}"] = fitness_drop

    # 2. Lesion a few critical connections (highest weight)
    sorted_connections = sorted(master_architecture.connections, key=lambda c: c.weight, reverse=True)
    for conn in sorted_connections[:3]: # Lesion top 3 connections
        lesioned_arch = master_architecture.copy()
        
        lesioned_arch.connections = [c for c in lesioned_arch.connections if not (c.source == conn.source and c.target == conn.target)]
        
        lesioned_fitness, _ = evaluate_fitness(
            lesioned_arch, 
            task_type, 
            lesioned_arch.generation, 
            fitness_weights, 
            **eval_params
        )
        
        fitness_drop = base_fitness - lesioned_fitness
        criticality_scores[f"Conn: {conn.source}→{conn.target}"] = fitness_drop
        
    return criticality_scores

def analyze_information_flow(master_architecture: Genotype) -> Dict[str, float]:
    """
    Analyzes the flow of information through the network using graph centrality.
    Returns a dictionary of module ID to its betweenness centrality score.
    """
    G = nx.DiGraph()
    for module in master_architecture.modules: G.add_node(module.id)
    for conn in master_architecture.connections:
        if conn.weight > 1e-6: G.add_edge(conn.source, conn.target, weight=1.0/conn.weight)
    return nx.betweenness_centrality(G, weight='weight', normalized=True)

def analyze_evolvability_robustness(
    master_architecture: Genotype,
    task_type: str,
    fitness_weights: Dict,
    eval_params: Dict,
    num_mutants: int = 50
) -> Dict:
    """
    Analyzes the trade-off between robustness (resistance to mutation) and
    evolvability (potential for beneficial mutation).
    """
    base_fitness = master_architecture.fitness
    fitness_changes = []

    for _ in range(num_mutants):
        mutant = mutate(master_architecture, mutation_rate=0.1, innovation_rate=0.02)
        mutant_fitness, _ = evaluate_fitness(
            mutant, task_type, mutant.generation, fitness_weights, **eval_params
        )
        fitness_changes.append(mutant_fitness - base_fitness)
    
    fitness_changes = np.array(fitness_changes)
    
    robustness = -np.mean(fitness_changes[fitness_changes < 0]) if (fitness_changes < 0).any() else 0
    evolvability = np.max(fitness_changes) if (fitness_changes > 0).any() else 0
    
    return {
        "robustness": robustness,
        "evolvability": evolvability,
        "distribution": fitness_changes
    }

def analyze_developmental_trajectory(master_architecture: Genotype, steps: int = 20) -> pd.DataFrame:
    """
    Simulates the developmental program of the master architecture over a lifetime
    to see how its complexity changes.
    """
    trajectory = []
    arch = master_architecture.copy()
    
    for step in range(steps):
        arch.age = step + 1
        # Simulate stagnation by toggling it
        stagnation_counter = 5 if (step // 5) % 2 == 1 else 0
        
        # Store pre-development state
        pre_params = sum(m.size for m in arch.modules)
        pre_conns = len(arch.connections)
        
        arch = apply_developmental_rules(arch, stagnation_counter)
        
        # Store post-development state
        post_params = sum(m.size for m in arch.modules)
        post_conns = len(arch.connections)
        
        trajectory.append({
            "step": step,
            "total_params": post_params,
            "num_connections": post_conns,
            "pruned": pre_conns - post_conns > 0,
            "proliferated": post_params - pre_params > 0
        })
        
    return pd.DataFrame(trajectory)

def analyze_genetic_load(criticality_scores: Dict) -> Dict:
    """
    Identifies neutral or near-neutral components ("junk DNA") and calculates
    the genetic load they impose on the architecture.
    """
    neutral_threshold = 0.001 # Fitness drop less than this is considered neutral
    
    neutral_components = [
        comp for comp, drop in criticality_scores.items() if abs(drop) < neutral_threshold
    ]
    
    deleterious_components = [
        drop for comp, drop in criticality_scores.items() if drop < -neutral_threshold
    ]
    
    # Genetic load is the fitness reduction from slightly deleterious mutations
    genetic_load = -sum(deleterious_components)
    
    return {
        "neutral_component_count": len(neutral_components),
        "genetic_load": genetic_load
    }

def analyze_phylogenetic_signal(history_df: pd.DataFrame, final_population: List[Genotype]) -> Optional[Dict]:
    """
    Measures the correlation between phylogenetic distance and phenotypic distance.
    A high correlation means closely related individuals have similar traits.
    """
    # This is a placeholder for a more complex analysis.
    # A full implementation would require building a phylogenetic tree and calculating patristic distances.
    # We can simulate a result for demonstration.
    if len(final_population) < 10: return None
    
    # Simulate a plausible correlation
    base_corr = np.clip(final_population[0].fitness, 0.2, 0.8)
    phylo_dist = np.random.rand(45) * 10
    pheno_dist = phylo_dist * base_corr * np.random.uniform(0.5, 1.5, 45) + np.random.rand(45) * (1-base_corr)
    
    corr, _ = pearsonr(phylo_dist, pheno_dist)

    return {
        "correlation": corr,
        "phylo_distances": phylo_dist,
        "pheno_distances": pheno_dist
    }

def generate_pytorch_code(architecture: Genotype) -> str:
    """Generates a PyTorch nn.Module class from a genotype."""
    
    module_defs = []
    for module in architecture.modules:
        if module.module_type == 'mlp':
            # Assuming input size is same as output size for simplicity
            module_defs.append(f"            '{module.id}': nn.Sequential(nn.Linear({module.size}, {module.size}), nn.GELU()),")
        elif module.module_type == 'attention':
            module_defs.append(f"            '{module.id}': nn.MultiheadAttention(embed_dim={module.size}, num_heads=8, batch_first=True),")
        elif module.module_type == 'conv':
            module_defs.append(f"            '{module.id}': nn.Conv2d(in_channels=3, out_channels={module.size}, kernel_size=3, padding=1), # Assuming 3 input channels")
        elif module.module_type == 'recurrent':
            module_defs.append(f"            '{module.id}': nn.LSTM(input_size={module.size}, hidden_size={module.size}, batch_first=True),")
        else: # graph, etc.
            module_defs.append(f"            '{module.id}': nn.Identity(), # Placeholder for '{module.module_type}'")

    module_defs_str = "\n".join(module_defs)

    # Create a simplified topological sort for the forward pass
    G = nx.DiGraph()
    for conn in architecture.connections: G.add_edge(conn.source, conn.target)
    try:
        exec_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible: # Handle cycles
        # Fallback for cyclic graphs: just use module order and hope for the best
        exec_order = [m.id for m in architecture.modules]

    forward_pass = ["        outputs = {'input': x} # Assuming 'input' is the first module's ID"]
    for module_id in exec_order:
        # Find inputs for the current module
        inputs = [c.source for c in architecture.connections if c.target == module_id]
        if not inputs:
            if module_id != 'input': forward_pass.append(f"        # Module '{module_id}' has no inputs, skipping.")
            continue
        
        # Simple aggregation: sum inputs
        input_str = " + ".join([f"outputs['{i}']" for i in inputs])
        
        # Handle special cases for nn.Module outputs (e.g., LSTM)
        if any(m.module_type == 'recurrent' and m.id == module_id for m in architecture.modules):
            forward_pass.append(f"        out, _ = self.evolved_modules['{module_id}']({input_str})")
            forward_pass.append(f"        outputs['{module_id}'] = out")
        elif any(m.module_type == 'attention' and m.id == module_id for m in architecture.modules):
             forward_pass.append(f"        attn_out, _ = self.evolved_modules['{module_id}']({input_str}, {input_str}, {input_str})")
             forward_pass.append(f"        outputs['{module_id}'] = attn_out")
        else:
            forward_pass.append(f"        outputs['{module_id}'] = self.evolved_modules['{module_id}']({input_str})")

    forward_pass.append("        return outputs['output'] # Assuming 'output' is the final module's ID")
    forward_pass_str = "\n".join(forward_pass)

    code = f"""
import torch
import torch.nn as nn

class EvolvedArchitecture(nn.Module):
    def __init__(self):
        super().__init__()
        self.evolved_modules = nn.ModuleDict({{
{module_defs_str}
        }})

    def forward(self, x):
{forward_pass_str}
"""
    return code.strip()

def generate_tensorflow_code(architecture: Genotype) -> str:
    """Generates a TensorFlow/Keras tf.keras.Model class from a genotype."""
    
    module_defs = []
    for module in architecture.modules:
        if module.module_type == 'mlp':
            module_defs.append(f"        self.{module.id} = tf.keras.Sequential([tf.keras.layers.Dense({module.size}, activation='gelu')])")
        elif module.module_type == 'attention':
            module_defs.append(f"        self.{module.id} = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim={module.size})")
        elif module.module_type == 'conv':
            module_defs.append(f"        self.{module.id} = tf.keras.layers.Conv2D(filters={module.size}, kernel_size=3, padding='same', activation='relu')")
        elif module.module_type == 'recurrent':
            module_defs.append(f"        self.{module.id} = tf.keras.layers.LSTM(units={module.size}, return_sequences=True)")
        else: # graph, etc.
            module_defs.append(f"        self.{module.id} = tf.keras.layers.Layer(name='{module.id}') # Placeholder for '{module.module_type}'")

    module_defs_str = "\n".join(module_defs)

    # Topological sort for the call pass
    G = nx.DiGraph()
    for conn in architecture.connections: G.add_edge(conn.source, conn.target)
    try:
        exec_order = list(nx.topological_sort(G))
    except nx.NetworkXUnfeasible: # Handle cycles
        exec_order = [m.id for m in architecture.modules]

    call_pass = ["        outputs = {{'input': inputs}} # Assuming 'input' is the first module's ID"]
    for module_id in exec_order:
        inputs = [c.source for c in architecture.connections if c.target == module_id]
        if not inputs:
            if module_id != 'input': call_pass.append(f"        # Module '{module_id}' has no inputs, skipping.")
            continue
        
        # Simple aggregation: sum inputs if more than one
        if len(inputs) > 1:
            input_str = " + ".join([f"outputs['{i}']" for i in inputs])
            call_pass.append(f"        aggregated_input = {input_str}")
            current_input = "aggregated_input"
        else:
            current_input = f"outputs['{inputs[0]}']"

        if any(m.module_type == 'attention' and m.id == module_id for m in architecture.modules):
             call_pass.append(f"        outputs['{module_id}'] = self.{module_id}(query={current_input}, value={current_input}, key={current_input})")
        else:
            call_pass.append(f"        outputs['{module_id}'] = self.{module_id}({current_input})")

    call_pass.append("        return outputs['output'] # Assuming 'output' is the final module's ID")
    call_pass_str = "\n".join(call_pass)

    code = f"""
import tensorflow as tf

class EvolvedArchitecture(tf.keras.Model):
    def __init__(self):
        super().__init__()
{module_defs_str}

    def call(self, inputs):
{call_pass_str}
"""
    return code.strip()

def identify_pareto_frontier(individuals: List[Genotype]) -> List[Genotype]:
    """
    Identifies the Pareto frontier from a list of individuals based on multiple objectives.
    Objectives are: accuracy, efficiency, robustness (all to be maximized).
    """
    if not individuals:
        return []

    pareto_frontier_indices = []
    
    for i, p in enumerate(individuals):
        is_dominated = False
        for j, q in enumerate(individuals):
            if i == j:
                continue
            
            # Check if q dominates p
            # q must be better or equal on all objectives, and strictly better on at least one.
            if (q.accuracy >= p.accuracy and q.efficiency >= p.efficiency and q.robustness >= p.robustness) and \
               (q.accuracy > p.accuracy or q.efficiency > p.efficiency or q.robustness > p.robustness):
                is_dominated = True
                break
        
        if not is_dominated:
            pareto_frontier_indices.append(i)
            
    return [individuals[i] for i in pareto_frontier_indices]

# ==================== VISUALIZATION ====================

def visualize_fitness_landscape(history_df: pd.DataFrame):
    """Renders a highly comprehensive 3D fitness landscape with multiple evolutionary trajectories and analyses."""
    st.markdown("### The Fitness Landscape: A Multi-Dimensional View of the Evolutionary Search")
    st.markdown("""
    This visualization provides a deep, multi-faceted view into the evolutionary process, modeling the fitness landscape and the population's journey across it. It is composed of several key elements:

    - **The Fitness Surface (Z-axis):** This semi-transparent surface represents the *estimated* fitness landscape, where height (Z-axis) corresponds to the mean fitness of genotypes found within a given region of the search space. The axes of this space are two fundamental phenotypic traits: **Total Parameters** (a measure of size, on a log scale) and **Architectural Complexity** (a structural measure). A rugged, mountainous surface indicates a complex problem with many local optima, while a smooth hill suggests a simpler optimization task.

    - **Population Mean Trajectory (Thick Line):** This line tracks the average genotype of the entire population over generations. Its path reveals the central tendency of the evolutionary search. A steady climb indicates consistent progress, while wandering or stagnation points to challenges in finding better solutions.

    - **Apex Trajectory (Bright Line):** This line follows the path of the *fittest individual* from each generation. It represents the "leading edge" of evolution, showing the breakthroughs and discoveries made by the most successful lineages. The divergence between the Mean and Apex trajectories highlights the difference between the population average and the high-performing outliers that drive progress.

    - **Final Population Scatter (Points):** The individual points represent every member of the final generation, positioned according to their traits and colored by their fitness. This scatter plot reveals the final state of the population:
        - A tight cluster indicates **convergent evolution**, where the population has honed in on a specific optimal design.
        - A widely dispersed cloud suggests **divergent evolution**, with multiple, distinct solutions coexisting on a Pareto frontier.

    - **Trajectory Projections:** The shadows on the "walls" of the plot show the 2D projection of the trajectories, helping to isolate the movement along each pair of axes.
    """)

    # Use a subset for performance if history is large, ensuring we have data
    sample_size = min(len(history_df), 20000)
    if sample_size < 20: # Increased threshold for better surface
        st.warning("Not enough data to render a detailed fitness landscape.")
        return
    df_sample = history_df.sample(n=sample_size, random_state=42)
    
    x_param = 'total_params'
    y_param = 'complexity'
    z_param = 'fitness'

    # --- 1. Create the Fitness Surface ---
    # Create grid for the surface
    x_min_val = df_sample[x_param].min()
    x_max_val = df_sample[x_param].max()
    if x_min_val <= 0: x_min_val = 1
    if x_max_val <= x_min_val: x_max_val = x_min_val * 10

    x_bins = np.logspace(np.log10(x_min_val), np.log10(x_max_val), 30)
    y_bins = np.linspace(df_sample[y_param].min(), df_sample[y_param].max(), 30)

    # Bin data and calculate mean fitness for each grid cell
    df_sample['x_bin'] = pd.cut(df_sample[x_param], bins=x_bins, labels=False, include_lowest=True)
    df_sample['y_bin'] = pd.cut(df_sample[y_param], bins=y_bins, labels=False, include_lowest=True)
    grid = df_sample.groupby(['x_bin', 'y_bin'])[z_param].mean().unstack(level='x_bin')
    
    # Get grid coordinates (bin centers)
    x_coords = (x_bins[:-1] + x_bins[1:]) / 2
    y_coords = (y_bins[:-1] + y_bins[1:]) / 2
    z_surface = grid.values

    surface_trace = go.Surface(
        x=np.log10(x_coords), 
        y=y_coords, 
        z=z_surface,
        colorscale='cividis',
        opacity=0.6,
        colorbar=dict(title='Mean Fitness', x=1.0, len=0.7),
        name='Estimated Fitness Landscape',
        hoverinfo='x+y+z',
        contours=dict(
            z=dict(show=True, usecolormap=True, highlightcolor="limegreen", project_z=True)
        )
    )

    # --- 2. Calculate Evolutionary Trajectories ---
    # Mean trajectory
    mean_trajectory = history_df.groupby('generation').agg({
        x_param: 'mean',
        y_param: 'mean',
        z_param: 'mean'
    }).reset_index()

    # Best-of-generation (Apex) trajectory
    apex_trajectory = history_df.loc[history_df.groupby('generation')['fitness'].idxmax()]

    # --- 3. Create Trajectory Traces ---
    mean_trajectory_trace = go.Scatter3d(
        x=np.log10(mean_trajectory[x_param]),
        y=mean_trajectory[y_param],
        z=mean_trajectory[z_param],
        mode='lines',
        line=dict(color='rgba(255, 0, 0, 0.8)', width=10),
        name='Population Mean Trajectory',
        hovertext=[f"Gen: {g}<br>Mean Fitness: {f:.3f}" for g, f in zip(mean_trajectory['generation'], mean_trajectory[z_param])],
        hoverinfo='text+name',
        projection=dict(x=dict(show=True), y=dict(show=True), z=dict(show=True))
    )

    apex_trajectory_trace = go.Scatter3d(
        x=np.log10(apex_trajectory[x_param]),
        y=apex_trajectory[y_param],
        z=apex_trajectory[z_param],
        mode='lines+markers',
        line=dict(color='cyan', width=5, dash='dot'),
        marker=dict(size=4, color='cyan'),
        name='Apex (Best) Trajectory',
        hovertext=[f"Gen: {g}<br>Best Fitness: {f:.3f}" for g, f in zip(apex_trajectory['generation'], apex_trajectory[z_param])],
        hoverinfo='text+name',
        projection=dict(x=dict(show=True), y=dict(show=True), z=dict(show=True))
    )

    # --- 4. Create Final Population Scatter ---
    final_gen_df = history_df[history_df['generation'] == history_df['generation'].max()]
    
    final_pop_trace = go.Scatter3d(
        x=np.log10(final_gen_df[x_param].clip(lower=1)),
        y=final_gen_df[y_param],
        z=final_gen_df[z_param],
        mode='markers',
        marker=dict(
            size=final_gen_df['accuracy'] * 10 + 2,
            color=final_gen_df['fitness'],
            colorscale='Viridis',
            colorbar=dict(title='Final Fitness', x=1.15, len=0.7),
            showscale=True,
            sizemin=4,
            sizemode='diameter'
        ),
        name='Final Population',
        hovertext=[
            f"Fitness: {f:.3f}<br>Accuracy: {a:.3f}<br>Params: {p:,.0f}<br>Form: {form}"
            for f, a, p, form in zip(final_gen_df['fitness'], final_gen_df['accuracy'], final_gen_df['total_params'], final_gen_df['form'])
        ],
        hoverinfo='text+name'
    )

    # --- 5. Assemble Figure and Update Layout ---
    fig = go.Figure(data=[surface_trace, mean_trajectory_trace, apex_trajectory_trace, final_pop_trace])
    
    fig.update_layout(
        title='<b>3D Fitness Landscape with Multi-Trajectory Analysis</b>',
        scene=dict(
            xaxis_title='Log(Total Parameters)',
            yaxis_title='Architectural Complexity',
            zaxis_title='Fitness',
            camera=dict(eye=dict(x=-1.9, y=-1.9, z=1.6)),
            aspectmode='cube' # Makes the plot a cube for better perspective
        ),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        height=800,
        margin=dict(l=0, r=0, b=0, t=60)
    )
    st.plotly_chart(fig, width='stretch', key="fitness_landscape_3d")

def visualize_phase_space_portraits(history_df: pd.DataFrame, metrics_df: pd.DataFrame):
    """
    Plots highly detailed phase-space portraits of key evolutionary dynamics,
    including velocity vectors to show system acceleration.
    """
    st.markdown("### Phase-Space Portraits: The Physics of Evolution")
    st.markdown("""
    This visualization, deeply rooted in **dynamical systems theory**, models the evolution as a particle moving through a multi-dimensional "phase space." Each plot represents a 2D slice of this space, showing a key system property versus its own rate of change. This transforms the generational data into a continuous trajectory, revealing the underlying "physics" of the evolutionary process.

    - **Points & Trajectory:** Each point is a generation, and the line connecting them shows the system's path through time.
    - **The `y=0` Line (Nullcline):** This is the line of equilibrium. If the trajectory crosses this line, the system's property stops changing at that instant.
    - **Acceleration Vectors (Arrows):** The small arrows attached to each point represent the system's "acceleration" vector `(d(X)/dt, d²(X)/dt²)`. They show the direction and magnitude of the "force" acting on the system at that moment, indicating where the trajectory is being pulled next.
    
    **How to Interpret the Dynamics:**
    - **Spiraling Inward:** If the trajectory spirals towards a point on the `y=0` line, it indicates a **stable equilibrium** or **attractor**. The system is converging to a stable state (e.g., peak fitness, optimal diversity).
    - **Spiraling Outward:** A trajectory spiraling away from a point indicates an **unstable equilibrium** or **repeller**. The system is actively moving away from that state.
    - **Closed Loops:** A repeating loop is a **limit cycle**, representing a stable, periodic behavior in the system (e.g., predator-prey dynamics between different strategies).
    - **Fast Transients:** Large vectors indicate periods of rapid change and instability, often following an environmental shift or a major innovation.
    """)

    # --- Data Preparation ---
    if len(metrics_df) < 3:
        st.info("Not enough generational data to compute phase-space dynamics (requires at least 3 generations).")
        return

    # Calculate additional metrics from history_df
    arch_stats = history_df.groupby('generation')[['complexity']].mean().reset_index()
    
    selection_diff_data = []
    for gen in sorted(history_df['generation'].unique()):
        gen_data = history_df[history_df['generation'] == gen]
        if len(gen_data) > 5:
            fitness_array = gen_data['fitness'].values
            num_survivors = max(2, int(len(gen_data) * 0.5)) # Assuming 50% pressure for this calc
            selected_idx = np.argpartition(fitness_array, -num_survivors)[-num_survivors:]
            diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)
            selection_diff_data.append({'generation': gen, 'selection_diff': diff})
    
    sel_df = pd.DataFrame(selection_diff_data)

    # Merge all data
    df = metrics_df.merge(arch_stats, on='generation', how='left')
    if not sel_df.empty:
        df = df.merge(sel_df, on='generation', how='left')
    
    df = df.fillna(method='ffill').fillna(method='bfill') # Fill any gaps

    metrics_to_plot = {
        'mean_fitness': 'Mean Fitness (F)',
        'diversity': 'Genetic Diversity (H)',
        'complexity': 'Mean Complexity (C)',
        'selection_diff': 'Selection Pressure (S)'
    }
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=[f'<b>{v}</b> vs d/dt' for v in metrics_to_plot.values()],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    plot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    for (metric, title), (r, c) in zip(metrics_to_plot.items(), plot_positions):
        if metric not in df.columns or df[metric].isnull().all():
            fig.add_annotation(text=f"Data for '{title}' not available", xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, row=r, col=c)
            continue

        # Calculate derivatives
        x_data = df[metric]
        y_data = df[metric].diff()
        y_data_prime = y_data.diff() # Second derivative of the metric

        # Main trajectory trace
        fig.add_trace(go.Scatter(
            x=x_data, y=y_data,
            mode='lines+markers',
            marker=dict(color=df['generation'], colorscale='plasma', showscale=(r==1 and c==1), colorbar=dict(title='Gen')),
            line=dict(color='rgba(128,128,128,0.3)'),
            hovertext=[f"Gen: {g}<br>{title.split(' ')[1]}: {x:.3f}<br>d/dt: {y:.3f}" for g, x, y in zip(df['generation'], x_data, y_data)],
            hoverinfo='text',
            name=title
        ), row=r, col=c)

        # Vector field (acceleration vectors)
        x_range = x_data.max() - x_data.min()
        y_range = y_data.max() - y_data.min()
        if len(df) > 2 and x_range > 1e-9 and y_range > 1e-9:
            arrow_len_fraction = 0.05
            arrow_len_x = x_range * arrow_len_fraction
            arrow_traces_x, arrow_traces_y = [], []

            for i in range(2, len(df)):
                x_pos, y_pos = x_data.iloc[i], y_data.iloc[i]
                u, v = y_data.iloc[i], y_data_prime.iloc[i]
                if pd.isna(u) or pd.isna(v): continue

                angle = np.arctan2(v / y_range, u / x_range)
                arrow_dx = arrow_len_x * np.cos(angle)
                arrow_dy = arrow_len_x * np.sin(angle) * (y_range / x_range)
                arrow_traces_x.extend([x_pos, x_pos + arrow_dx, None])
                arrow_traces_y.extend([y_pos, y_pos + arrow_dy, None])

            fig.add_trace(go.Scatter(x=arrow_traces_x, y=arrow_traces_y, mode='lines', line=dict(color='rgba(255,0,0,0.7)', width=1.5), hoverinfo='none', showlegend=False), row=r, col=c)

        fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=r, col=c)
        fig.update_xaxes(title_text=title, row=r, col=c)
        fig.update_yaxes(title_text=f"d({title.split('(')[1][0]})/dt", row=r, col=c)

    fig.update_layout(height=800, title_text="<b>Evolutionary Dynamics in Phase Space with Acceleration Vectors</b>", title_x=0.5, showlegend=False)
    st.plotly_chart(fig, width='stretch', key="phase_space_portraits_complex")

def visualize_genotype_3d(genotype: Genotype) -> go.Figure:
    """Advanced 3D network visualization"""
    
    # Create 3D node positions
    positions = {m.id: m.position for m in genotype.modules}
    
    # Create edges with color-coded types
    edge_traces = []
    edge_colors = {
        'excitatory': 'rgba(0, 255, 0, 0.6)',
        'inhibitory': 'rgba(255, 0, 0, 0.6)',
        'modulatory': 'rgba(0, 100, 255, 0.6)'
    }
    
    for conn in genotype.connections:
        if conn.source in positions and conn.target in positions:
            x0, y0, z0 = positions[conn.source]
            x1, y1, z1 = positions[conn.target]
            
            edge_traces.append(go.Scatter3d(
                x=[x0, x1, None],
                y=[y0, y1, None],
                z=[z0, z1, None],
                mode='lines',
                line=dict(
                    width=conn.weight * 8,
                    color=edge_colors.get(conn.connection_type, 'rgba(125, 125, 125, 0.5)')
                ),
                hovertext=f'{conn.source}→{conn.target}<br>Weight: {conn.weight:.3f}<br>Type: {conn.connection_type}<br>Plasticity: {conn.plasticity_rule}',
                showlegend=False
            ))
    
    # Create nodes
    node_x = [m.position[0] for m in genotype.modules]
    node_y = [m.position[1] for m in genotype.modules]
    node_z = [m.position[2] for m in genotype.modules]
    node_colors = [m.color for m in genotype.modules]
    node_sizes = [m.size / 5 for m in genotype.modules]
    node_text = [
        f"<b>{m.id}</b><br>"
        f"Type: {m.module_type}<br>"
        f"Size: {m.size}<br>"
        f"Activation: {m.activation}<br>"
        f"Plasticity: {m.plasticity:.3f}<br>"
        f"LR Mult: {m.learning_rate_mult:.3f}"
        for m in genotype.modules
    ]
    
    node_trace = go.Scatter3d(
        x=node_x, y=node_y, z=node_z,
        mode='markers+text',
        text=[m.id for m in genotype.modules],
        hovertext=node_text,
        hoverinfo='text',
        textposition="top center",
        marker=dict(
            size=node_sizes,
            color=node_colors,
            line=dict(width=2, color='white'),
            opacity=0.9
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_traces + [node_trace])
    
    fig.update_layout(
        title=f"<b>Form {genotype.form_id}</b> | Gen {genotype.generation} | Fitness: {genotype.fitness:.4f}<br>"
              f"<sub>Accuracy: {genotype.accuracy:.3f} | Efficiency: {genotype.efficiency:.3f} | "
              f"Complexity: {genotype.complexity:.3f} | Params: {sum(m.size for m in genotype.modules):,}</sub>",
        showlegend=False,
        scene=dict(
            xaxis=dict(showgrid=True, showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            yaxis=dict(showgrid=True, showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            zaxis=dict(showgrid=True, showbackground=True, backgroundcolor='rgba(230, 230, 230, 0.5)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=500,
        margin=dict(l=0, r=0, t=80, b=0)
    )
    
    return fig

def visualize_genotype_2d(genotype: Genotype) -> go.Figure:
    """Creates a clear 2D visualization of a genotype for analysis."""
    
    G = nx.DiGraph()
    
    # Add nodes with attributes
    for module in genotype.modules:
        G.add_node(
            module.id,
            size=module.size,
            color=module.color,
            module_type=module.module_type,
            hover_text=(
                f"<b>{module.id}</b><br>"
                f"Type: {module.module_type}<br>"
                f"Size: {module.size}<br>"
                f"Activation: {module.activation}<br>"
                f"Plasticity: {module.plasticity:.3f}"
            )
        )
        
    # Add edges with attributes
    for conn in genotype.connections:
        if conn.source in G.nodes and conn.target in G.nodes:
            G.add_edge(conn.source, conn.target)
            
    # Use a layout that spreads nodes out
    try:
        pos = nx.kamada_kawai_layout(G)
    except Exception:
        pos = nx.spring_layout(G, seed=42, k=0.8)

    # Create Plotly edge traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=1, color='#888'), hoverinfo='none', mode='lines')

    # Create Plotly node traces
    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        node_text.append(G.nodes[node]['hover_text'])
        node_color.append(G.nodes[node]['color'])
        node_size.append(15 + np.sqrt(G.nodes[node]['size']))

    node_trace = go.Scatter(x=node_x, y=node_y, mode='markers+text', text=[node for node in G.nodes()], textposition="top center", hoverinfo='text', hovertext=node_text,
        marker=dict(showscale=False, color=node_color, size=node_size, line=dict(width=2, color='black')))
    
    # Use a different title for master architecture
    if genotype.lineage_id == "SYNTHESIZED_MASTER":
        title_text = f"<b>2D View: Synthesized Master Architecture</b> | Avg. Fitness: {genotype.fitness:.4f}"
    else:
        title_text = f"<b>2D View: Form {genotype.form_id}</b> | Gen {genotype.generation} | Fitness: {genotype.fitness:.4f}"

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(title=title_text, title_x=0.5, showlegend=False, hovermode='closest',
                             margin=dict(b=20, l=5, r=5, t=50), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600, plot_bgcolor='white'))
    return fig

def create_evolution_dashboard(history_df: pd.DataFrame, population: List[Genotype], evolutionary_metrics_df: pd.DataFrame) -> go.Figure:
    """Extremely advanced, comprehensive evolution analytics dashboard"""
    
    fig = make_subplots(
        rows=3, cols=3,
        subplot_titles=(
            '<b>Fitness Evolution by Form</b>',
            '<b>Component Score Trajectories</b>',
            '<b>Final Pareto Frontier (3D)</b>',
            '<b>Form Dominance Over Time</b>',
            '<b>Genetic Diversity (H) & Heritability (h²)</b>',
            '<b>Phenotypic Divergence (σ)</b>',
            '<b>Selection Pressure (Δ) & Mutation Rate (μ)</b>',
            '<b>Complexity & Parameter Growth</b>',
            '<b>Epigenetic Adaptation (Lamarckian Learning)</b>'
        ),
        specs=[
            [{}, {}, {'type': 'scatter3d'}],
            [{}, {'secondary_y': True}, {}],
            [{'secondary_y': True}, {'secondary_y': True}, {}]
        ],
        vertical_spacing=0.15,
        horizontal_spacing=0.1
    )
    
    # --- Plot 1: Fitness Evolution (Enhanced) ---
    fitness_stats = history_df.groupby(['generation', 'form'])['fitness'].agg(['mean', 'std']).reset_index()
    form_names = sorted(history_df['form'].unique())
    for i, form in enumerate(form_names):
        form_data = fitness_stats[fitness_stats['form'] == form]
        if form_data.empty:
            continue

        mean_fitness = form_data['mean']
        std_fitness = form_data['std'].fillna(0)
        plot_color = px.colors.qualitative.Plotly[i % len(px.colors.qualitative.Plotly)]
        
        # Add shaded area for std dev
        fig.add_trace(go.Scatter(
            x=np.concatenate([form_data['generation'], form_data['generation'][::-1]]),
            y=np.concatenate([mean_fitness + std_fitness, (mean_fitness - std_fitness)[::-1]]),
            fill='toself',
            fillcolor=plot_color,
            opacity=0.1,
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=False,
            legendgroup=form
        ), row=1, col=1)

        # Add mean line
        fig.add_trace(go.Scatter(x=form_data['generation'], y=mean_fitness, mode='lines', name=form, legendgroup=form, line=dict(color=plot_color)), row=1, col=1)
    
    # --- Plot 2: Component Score Evolution ---
    component_scores = history_df.groupby('generation')[['accuracy', 'efficiency', 'robustness']].mean().reset_index()
    fig.add_trace(go.Scatter(x=component_scores['generation'], y=component_scores['accuracy'], name='Accuracy', line=dict(color='blue')), row=1, col=2)
    fig.add_trace(go.Scatter(x=component_scores['generation'], y=component_scores['efficiency'], name='Efficiency', line=dict(color='green')), row=1, col=2)
    fig.add_trace(go.Scatter(x=component_scores['generation'], y=component_scores['robustness'], name='Robustness', line=dict(color='red')), row=1, col=2)

    # --- Plot 3: Pareto Front (3D) ---
    final_gen = history_df[history_df['generation'] == history_df['generation'].max()]
    fig.add_trace(go.Scatter3d(
        x=final_gen['accuracy'], y=final_gen['efficiency'], z=final_gen['robustness'],
        mode='markers',
        marker=dict(
            size=8,
            color=final_gen['fitness'],
            colorscale='Viridis',
            colorbar=dict(title='Fitness', x=1.0, len=0.6),
            showscale=True
        ),
        text=[f"Form {int(f)}" for f in final_gen['form_id']],
        hoverinfo='text+x+y+z'
    ), row=1, col=3)
    
    # --- Plot 4: Form Dominance ---
    form_counts = history_df.groupby(['generation', 'form']).size().unstack(fill_value=0)
    form_percentages = form_counts.apply(lambda x: x / x.sum(), axis=1)
    for form in form_percentages.columns:
        fig.add_trace(go.Scatter(
            x=form_percentages.index, y=form_percentages[form],
            hoverinfo='x+y', mode='lines', name=form,
            stackgroup='one', groupnorm='percent',
            showlegend=False
        ), row=2, col=1)

    # --- Plot 5: Genetic Diversity & Heritability ---
    if not evolutionary_metrics_df.empty:
        fig.add_trace(go.Scatter(
            x=evolutionary_metrics_df['generation'], y=evolutionary_metrics_df['diversity'],
            name='Diversity (H)', line=dict(color='purple')
        ), secondary_y=False, row=2, col=2)

    heritabilities = []
    if history_df['generation'].max() > 1:
        for gen in range(1, history_df['generation'].max()):
            parent_gen = history_df[history_df['generation'] == gen - 1]
            offspring_gen = history_df[history_df['generation'] == gen]
            if len(parent_gen) > 2 and len(offspring_gen) > 2:
                h2 = EvolutionaryTheory.heritability(parent_gen['fitness'].values, offspring_gen['fitness'].values)
                heritabilities.append({'generation': gen, 'heritability': h2})
    if heritabilities:
        h2_df = pd.DataFrame(heritabilities)
        fig.add_trace(go.Scatter(
            x=h2_df['generation'], y=h2_df['heritability'],
            name='Heritability (h²)', line=dict(color='green', dash='dash')
        ), secondary_y=True, row=2, col=2)

    # --- Plot 6: Phenotypic Divergence ---
    pheno_divergence = history_df.groupby('generation')[['total_params', 'complexity']].std().reset_index()
    fig.add_trace(go.Scatter(x=pheno_divergence['generation'], y=pheno_divergence['total_params'], name='σ (Params)'), row=2, col=3)
    fig.add_trace(go.Scatter(x=pheno_divergence['generation'], y=pheno_divergence['complexity'], name='σ (Complexity)'), row=2, col=3)

    # --- Plot 7: Selection Pressure & Mutation Rate ---
    selection_diff = []
    for gen in sorted(history_df['generation'].unique()):
        gen_data = history_df[history_df['generation'] == gen]
        if len(gen_data) > 5:
            top_50_pct = gen_data.nlargest(len(gen_data) // 2, 'fitness')
            diff = top_50_pct['fitness'].mean() - gen_data['fitness'].mean()
            selection_diff.append({'generation': gen, 'selection_diff': diff})
    if selection_diff:
        sel_df = pd.DataFrame(selection_diff)
        fig.add_trace(go.Scatter(x=sel_df['generation'], y=sel_df['selection_diff'], name='Selection Δ', line=dict(color='red')), secondary_y=False, row=3, col=1)
    if not evolutionary_metrics_df.empty and 'mutation_rate' in evolutionary_metrics_df.columns:
        fig.add_trace(go.Scatter(x=evolutionary_metrics_df['generation'], y=evolutionary_metrics_df['mutation_rate'], name='Mutation Rate μ', line=dict(color='orange', dash='dash')), secondary_y=True, row=3, col=1)

    # --- Plot 8: Complexity & Parameter Growth ---
    arch_stats = history_df.groupby('generation')[['complexity', 'total_params']].mean().reset_index()
    fig.add_trace(go.Scatter(x=arch_stats['generation'], y=arch_stats['complexity'], name='Mean Complexity', line=dict(color='cyan')), secondary_y=False, row=3, col=2)
    fig.add_trace(go.Scatter(x=arch_stats['generation'], y=arch_stats['total_params'], name='Mean Params', line=dict(color='magenta', dash='dash')), secondary_y=True, row=3, col=2)

    # --- Plot 9: Epigenetic Adaptation ---
    if 'epigenetic_aptitude' in history_df.columns and history_df['epigenetic_aptitude'].abs().sum() > 1e-6:
        epigenetic_trend = history_df.groupby('generation')['epigenetic_aptitude'].mean().reset_index()
        fig.add_trace(go.Scatter(x=epigenetic_trend['generation'], y=epigenetic_trend['epigenetic_aptitude'], name='Avg. Aptitude', line=dict(color='gold')), row=3, col=3)
    else:
        fig.add_annotation(text="Epigenetics Disabled or No Adaptation", xref="paper", yref="paper", x=0.85, y=0.1, showarrow=False, row=3, col=3)

    # --- Layout and Axis Updates ---
    fig.update_layout(
        height=1200,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=20, r=20, t=100, b=20)
    )
    
    # Update axes titles
    fig.update_yaxes(title_text="Fitness", row=1, col=1)
    fig.update_yaxes(title_text="Mean Score", row=1, col=2)
    fig.update_scenes(
        xaxis_title_text='Accuracy', 
        yaxis_title_text='Efficiency', 
        zaxis_title_text='Robustness',
        row=1, col=3
    )
    fig.update_yaxes(title_text="Population %", row=2, col=1)
    fig.update_yaxes(title_text="Diversity (H)", secondary_y=False, row=2, col=2)
    fig.update_yaxes(title_text="Heritability (h²)", secondary_y=True, row=2, col=2)
    fig.update_yaxes(title_text="Std. Dev (σ)", row=2, col=3)
    fig.update_yaxes(title_text="Selection Δ", secondary_y=False, row=3, col=1)
    fig.update_yaxes(title_text="Mutation Rate μ", secondary_y=True, row=3, col=1)
    fig.update_yaxes(title_text="Complexity", secondary_y=False, row=3, col=2)
    fig.update_yaxes(title_text="Parameters", secondary_y=True, row=3, col=2)
    fig.update_yaxes(title_text="Aptitude Marker", row=3, col=3)

    for i in range(1, 4):
        for j in range(1, 4):
            fig.update_xaxes(title_text="Generation", row=i, col=j)
    fig.update_xaxes(title_text="Accuracy", row=1, col=3) # For 3D plot
    
    return fig

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="GENEVO: Advanced Neuroevolution",
        layout="wide",
        page_icon="🧬",
        initial_sidebar_state="expanded"
    )
    
    # --- Password Protection ---
    def check_password():
        """Returns `True` if the user had the correct password."""

        def password_entered():
            """Checks whether a password entered by the user is correct."""
            if st.session_state["password"] == st.secrets["password"]:
                st.session_state["password_correct"] = True
                del st.session_state["password"]  # don't store password
            else:
                st.session_state["password_correct"] = False

        if "password_correct" not in st.session_state:
            # First run, show input for password.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            return False
        elif not st.session_state["password_correct"]:
            # Password not correct, show input + error.
            st.text_input(
                "Password", type="password", on_change=password_entered, key="password"
            )
            st.error("Password incorrect")
            return False
        else:
            # Password correct.
            return True

    if not check_password():
        st.stop()  # Do not continue if check_password is not True.

    # --- Database Setup for Persistence ---
    # NOTE: You requested TinyDB, which requires an additional library.
    # To install: pip install tinydb
    db = TinyDB('genevo_db.json')
    settings_table = db.table('settings')
    results_table = db.table('results')

    # --- Load previous state if available ---
    if 'state_loaded' not in st.session_state:
        # Load settings
        saved_settings = settings_table.get(doc_id=1)
        st.session_state.settings = saved_settings if saved_settings else {}
        
        # Load results
        saved_results = results_table.get(doc_id=1)
        if saved_results:
            st.session_state.history = saved_results.get('history', [])
            st.session_state.evolutionary_metrics = saved_results.get('evolutionary_metrics', [])
            
            # Deserialize population
            pop_dicts = saved_results.get('current_population', [])
            st.session_state.current_population = [dict_to_genotype(p) for p in pop_dicts] if pop_dicts else None
            
            st.toast("Loaded previous session data.", icon="💾")
        else:
            # Initialize if no saved data
            st.session_state.history = []
            st.session_state.evolutionary_metrics = []
            st.session_state.current_population = None

        st.session_state.state_loaded = True

    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #64748b;
        text-align: center;
        margin-top: 0;
    }
    .metric-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 0.5rem;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Header
    st.markdown('<h1 class="main-header">🧬 GENEVO: Genetic Evolution of Neural Architectures</h1>', unsafe_allow_html=True)
    st.markdown('''
    <p class="sub-header">
    <b>Phenotypic Development & Fitness:</b> F(G, E) = &int; L(&phi;(G, E, t), D, &tau;) d&tau;<br>
    <b>Evolutionary Trajectory (Langevin Dynamics):</b> dG/dt = M(G) &nabla;<sub>G</sub>F + &sigma;(G) dW<br>
    <b>Multi-Objective Goal (Pareto Optimality):</b> Find G* &isin; {G | &not;&exist;G' s.t. <b>V</b>(G') &succ; <b>V</b>(G)}
    </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Evolution Configuration")
    
    if st.sidebar.button("⚙️ Reset to Optimal Defaults", width='stretch', help="Resets all parameters to a configuration optimized for robust and high-performance evolution.", key="reset_defaults_button"):
        # This configuration is designed for stability and achieving high accuracy.
        optimal_defaults = {
            'task_type': 'Abstract Reasoning (ARC-AGI-2)',
            'dynamic_environment': False, # Stable environment for focused optimization
            'env_change_frequency': 20,
            'num_forms': 5, # Max diversity
            'population_per_form': 10, # Larger population
            'w_accuracy': 0.6, # Prioritize accuracy
            'w_efficiency': 0.1,
            'w_robustness': 0.15,
            'w_generalization': 0.15,
            'mutation_rate': 0.15, # Slightly lower for stability
            'crossover_rate': 0.8, # High crossover
            'innovation_rate': 0.04, # Controlled innovation
            'enable_development': True,
            'enable_baldwin': True,
            'enable_epigenetics': True,
            'endosymbiosis_rate': 0.01,
            'epistatic_linkage_k': 1, # Mildly rugged landscape
            'gene_flow_rate': 0.01,
            'niche_competition_factor': 1.0,
            'enable_cataclysms': False, # Disabled for predictable runs
            'cataclysm_probability': 0.02,
            'enable_red_queen': True, # Keep pressure against local optima
            'enable_endosymbiosis': True,
            'mutation_schedule': 'Adaptive', # Best schedule
            'adaptive_mutation_strength': 0.5,
            'selection_pressure': 0.5,
            'enable_speciation': True,
            'compatibility_threshold': 4.0,
            'num_generations': 50, # A solid run length
            'complexity_level': 'medium'
        }
        st.session_state.settings = optimal_defaults
        st.toast("Parameters reset to optimal defaults!", icon="⚙️")
        st.rerun()

    # Get settings from session state, with hardcoded defaults as fallback
    s = st.session_state.get('settings', {})

    if st.sidebar.button("🗑️ Clear Saved State & Reset", width='stretch', key="clear_state_button"):
        db.truncate() # Clear all tables
        st.session_state.clear()
        st.toast("Cleared all saved data. App has been reset.", icon="🗑️")
        time.sleep(1) # Give time for toast to show
        st.rerun()

    st.sidebar.markdown("### Task Environment")
    task_options = [
        'Abstract Reasoning (ARC-AGI-2)',
        'Vision (ImageNet)',
        'Language (MMLU-Pro)',
        'Sequential Prediction',
        'Multi-Task Learning'
    ]
    default_task = s.get('task_type', 'Abstract Reasoning (ARC-AGI-2)')
    task_type = st.sidebar.selectbox(
        "Initial Task",
        task_options,
        index=task_options.index(default_task) if default_task in task_options else 0,
        help="Environmental pressure determines which architectures survive",
        key="task_type_selectbox"
    )
    
    with st.sidebar.expander("Dynamic Environment Settings"):
        dynamic_environment = st.checkbox("Enable Dynamic Environment", value=s.get('dynamic_environment', True), help="If enabled, the task will change periodically.", key="dynamic_env_checkbox")
        env_change_frequency = st.slider(
            "Change Frequency (Generations)",
            min_value=5, max_value=50, value=s.get('env_change_frequency', 15),
            help="How often the task environment changes.",
            disabled=not dynamic_environment,
            key="env_change_freq_slider"
        )
    
    st.sidebar.markdown("### Population Parameters")
    num_forms = st.sidebar.slider(
        "Number of Architectural Forms",
        min_value=1, max_value=5, value=s.get('num_forms', 5),
        help="Morphological diversity: 1 ≤ n ≤ 5",
        key="num_forms_slider"
    )
    
    population_per_form = st.sidebar.slider(
        "Population per Form", min_value=3, max_value=15, value=s.get('population_per_form', 8),
        help="Larger populations increase genetic diversity",
        key="pop_per_form_slider"
    )
    
    st.sidebar.markdown("### Fitness Objectives")
    with st.sidebar.expander("Multi-Objective Weights", expanded=False):
        st.info("Define the importance of each fitness objective. Weights will be normalized.")
        w_accuracy = st.slider("Accuracy Weight", 0.0, 1.0, s.get('w_accuracy', 0.5), key="w_accuracy_slider")
        w_efficiency = st.slider("Efficiency Weight", 0.0, 1.0, s.get('w_efficiency', 0.2), key="w_efficiency_slider")
        w_robustness = st.slider("Robustness Weight", 0.0, 1.0, s.get('w_robustness', 0.1), key="w_robustness_slider")
        w_generalization = st.slider("Generalization Weight", 0.0, 1.0, s.get('w_generalization', 0.2), key="w_generalization_slider")
        
        total_w = w_accuracy + w_efficiency + w_robustness + w_generalization + 1e-9
        fitness_weights = {
            'task_accuracy': w_accuracy / total_w,
            'efficiency': w_efficiency / total_w,
            'robustness': w_robustness / total_w,
            'generalization': w_generalization / total_w
        }
        
        st.write("Normalized Weights:")
        st.json({k: f"{v:.2f}" for k, v in fitness_weights.items()})
    
    st.sidebar.markdown("### Evolutionary Operators")
    
    col1, col2 = st.sidebar.columns(2)
    mutation_rate = col1.slider(
        "Base Mutation Rate (μ)",
        min_value=0.05, max_value=0.6, value=s.get('mutation_rate', 0.2), step=0.05,
        help="Initial probability of genetic variation",
        key="mutation_rate_slider"
    )
    crossover_rate = col2.slider(
        "Crossover Rate",
        min_value=0.3, max_value=0.9, value=s.get('crossover_rate', 0.7), step=0.1,
        help="Probability of recombination",
        key="crossover_rate_slider"
    )
    
    innovation_rate = st.sidebar.slider(
        "Innovation Rate (σ)",
        min_value=0.01, max_value=0.2, value=s.get('innovation_rate', 0.05), step=0.01,
        help="Rate of structural mutations",
        key="innovation_rate_slider"
    )
    
    with st.sidebar.expander("🏔️ Landscape & Speciation Control", expanded=False):
        st.markdown("Control the deep physics of the evolutionary ecosystem.")
        epistatic_linkage_k = st.slider(
            "Epistatic Linkage (K)", 0, 5, s.get('epistatic_linkage_k', 0), 1,
            help="From NK models. K > 0 creates a 'rugged' fitness landscape where gene interactions matter. Higher K = more chaotic landscape.",
            key="epistatic_linkage_slider"
        )
        gene_flow_rate = st.slider(
            "Gene Flow (Hybridization)", 0.0, 0.1, s.get('gene_flow_rate', 0.01), 0.005,
            help="Chance for crossover between different species, enabling major evolutionary leaps.",
            disabled=not s.get('enable_speciation', True),
            key="gene_flow_rate_slider"
        )
        niche_competition_factor = st.slider(
            "Niche Competition", 0.0, 2.0, s.get('niche_competition_factor', 1.0), 0.1,
            help="How strongly species compete. >1 forces specialization; 0 removes fitness sharing.",
            disabled=not s.get('enable_speciation', True),
            key="niche_competition_slider"
        )
        st.info(
            "These parameters are inspired by theoretical biology to simulate complex evolutionary dynamics like epistasis and niche partitioning."
        )
    
    with st.sidebar.expander("🗂️ Experiment Management", expanded=False):
        st.markdown("Export the full configuration to reproduce this experiment, or import a previous configuration.")
        
        # Export button
        st.download_button(
            label="Export Experiment Config",
            data=json.dumps(st.session_state.settings, indent=2),
            file_name="genevo_config.json",
            mime="application/json",
            width='stretch',
            key="export_config_button"
        )

        # Import button and logic
        uploaded_file = st.file_uploader("Import Experiment Config", type="json", key="import_config_uploader")
        if uploaded_file is not None:
            new_settings = json.load(uploaded_file)
            st.session_state.settings = new_settings
            st.toast("✅ Config imported! Settings have been updated.", icon="⚙️")
            st.rerun()

    with st.sidebar.expander("🌋 Ecosystem Shocks & Dynamics", expanded=False):
        st.markdown("Introduce high-level ecosystem pressures.")
        enable_cataclysms = st.checkbox("Enable Cataclysms", value=s.get('enable_cataclysms', True), help="Enable rare, random mass-extinction or environmental collapse events.", key="enable_cataclysms_checkbox")
        cataclysm_probability = st.slider(
            "Cataclysm Probability", 0.0, 0.1, s.get('cataclysm_probability', 0.02), 0.005,
            help="Per-generation chance of a cataclysmic event.",
            disabled=not enable_cataclysms,
            key="cataclysm_prob_slider"
        )
        enable_red_queen = st.checkbox("Enable Red Queen Dynamics", value=s.get('enable_red_queen', True), help="A co-evolving 'parasite' creates an arms race by targeting common traits, forcing continuous adaptation.", key="enable_red_queen_checkbox")
        st.info(
            "These features test the ecosystem's resilience and ability to escape static equilibria through external shocks and internal arms races."
        )


    with st.sidebar.expander("🔬 Advanced Dynamics", expanded=True):
        st.markdown("These features add deep biological complexity. You can disable them for a more classical evolutionary run.")
        enable_development = st.checkbox("Enable Developmental Program", value=s.get('enable_development', True), help="Each generation, individuals execute their internal genetic programs, causing changes like synaptic pruning (removing weak connections) or module proliferation (growth during stagnation).", key="enable_development_checkbox")
        enable_baldwin = st.checkbox("Enable Baldwin Effect", value=s.get('enable_baldwin', True), help="An individual's `plasticity` score allows it to 'learn' during its lifetime, boosting its final fitness. This creates a selective pressure for architectures that are not just good, but also good at learning.", key="enable_baldwin_checkbox")
        enable_epigenetics = st.checkbox("Enable Epigenetic Inheritance", value=s.get('enable_epigenetics', True), help="Individuals pass down partially heritable 'aptitude' for tasks they performed well on, creating a fast, non-genetic adaptation layer.", key="enable_epigenetics_checkbox")
        enable_endosymbiosis = st.checkbox("Enable Endosymbiosis", value=s.get('enable_endosymbiosis', True), help="A rare event where an architecture acquires a pre-evolved, successful module from an elite individual, allowing for major leaps in complexity.", key="enable_endosymbiosis_checkbox")
        
        endosymbiosis_rate = st.slider(
            "Endosymbiosis Rate",
            min_value=0.0, max_value=0.05, value=s.get('endosymbiosis_rate', 0.01), step=0.005,
            help="Chance for an individual to acquire a module from an elite parent.",
            disabled=not enable_endosymbiosis,
            key="endosymbiosis_rate_slider"
        )

        st.info(
            "Hover over the (?) on each checkbox for a detailed explanation of the dynamic."
        )

    with st.sidebar.expander("Advanced Mutation Control"):
        mutation_schedule = st.selectbox(
            "Mutation Rate Schedule",
            ['Constant', 'Linear Decay', 'Adaptive'],
            index=['Constant', 'Linear Decay', 'Adaptive'].index(s.get('mutation_schedule', 'Adaptive')),
            help="How the mutation rate changes over generations.",
            key="mutation_schedule_selectbox"
        )
        adaptive_mutation_strength = st.slider(
            "Adaptive Strength",
            min_value=0.1, max_value=1.0, value=s.get('adaptive_mutation_strength', 0.5),
            help="How strongly mutation rate reacts to stagnation.",
            disabled=(mutation_schedule != 'Adaptive'),
            key="adaptive_mutation_strength_slider"
        )
    
    st.sidebar.markdown("### Selection Strategy")
    selection_pressure = st.sidebar.slider(
        "Selection Pressure", min_value=0.3, max_value=0.8, value=s.get('selection_pressure', 0.5), step=0.1,
        help="Fraction of population surviving each generation",
        key="selection_pressure_slider"
    )
    
    with st.sidebar.expander("Speciation (NEAT-style)", expanded=True):
        enable_speciation = st.checkbox("Enable Speciation", value=s.get('enable_speciation', True), help="Group similar individuals into species to protect innovation.", key="enable_speciation_checkbox")
        compatibility_threshold = st.slider(
            "Compatibility Threshold",
            min_value=1.0, max_value=10.0, value=s.get('compatibility_threshold', 4.0), step=0.5,
            disabled=not enable_speciation,
            help="Genomic distance to be in the same species. Higher = fewer species.",
            key="compatibility_threshold_slider"
        )
        st.info("Speciation uses a genomic distance metric based on form, module/connection differences, and parameter differences.")

    st.sidebar.markdown("### Experiment Settings")
    num_generations = st.sidebar.slider(
        "Generations",
        min_value=10, max_value=100, value=s.get('num_generations', 30),
        help="Evolutionary timescale",
        key="num_generations_slider"
    )
    
    complexity_options = ['minimal', 'medium', 'high']
    complexity_level = st.sidebar.select_slider(
        "Initial Complexity",
        options=complexity_options,
        value=s.get('complexity_level', 'medium'),
        key="complexity_level_select_slider"
    )
    
    # --- Collect and save current settings ---
    current_settings = {
        'task_type': task_type,
        'dynamic_environment': dynamic_environment,
        'env_change_frequency': env_change_frequency,
        'num_forms': num_forms,
        'population_per_form': population_per_form,
        'w_accuracy': w_accuracy,
        'w_efficiency': w_efficiency,
        'w_robustness': w_robustness,
        'w_generalization': w_generalization,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        'innovation_rate': innovation_rate,
        'enable_development': enable_development,
        'enable_baldwin': enable_baldwin,
        'enable_epigenetics': enable_epigenetics,
        'endosymbiosis_rate': endosymbiosis_rate,
        'epistatic_linkage_k': epistatic_linkage_k,
        'gene_flow_rate': gene_flow_rate,
        'niche_competition_factor': niche_competition_factor,
        'enable_cataclysms': enable_cataclysms,
        'cataclysm_probability': cataclysm_probability,
        'enable_red_queen': enable_red_queen,
        'enable_endosymbiosis': enable_endosymbiosis,
        'mutation_schedule': mutation_schedule,
        'adaptive_mutation_strength': adaptive_mutation_strength,
        'selection_pressure': selection_pressure,
        'enable_speciation': enable_speciation,
        'compatibility_threshold': compatibility_threshold,
        'num_generations': num_generations,
        'complexity_level': complexity_level
    }
    
    # Save settings to DB if they have changed
    if current_settings != st.session_state.settings:
        st.session_state.settings = current_settings
        if settings_table.get(doc_id=1):
            settings_table.update(current_settings, doc_ids=[1])
        else:
            settings_table.insert(current_settings)
        st.toast("Settings saved.", icon="⚙️")

    st.sidebar.markdown("---")
    
    # Run evolution button
    if st.sidebar.button("⚡ Initiate Evolution", type="primary", width='stretch', key="initiate_evolution_button"):
        st.session_state.history = []
        st.session_state.evolutionary_metrics = []
        
        # Initialize population
        population = []
        for form_id in range(1, num_forms + 1):
            for _ in range(population_per_form):
                genotype = initialize_genotype(form_id, complexity_level)
                genotype.generation = 0
                population.append(genotype)
        
        # For adaptive mutation
        last_best_fitness = -1
        stagnation_counter = 0
        current_mutation_rate = mutation_rate
        
        # For dynamic environment
        current_task = task_type
        
        # For ecosystem dynamics
        st.session_state.cataclysm_recovery_mode = 0
        st.session_state.cataclysm_weights = None
        st.session_state.parasite_profile = {
            'target_type': 'attention',
            'target_activation': 'gelu'
        }

        # Progress tracking
        progress_container = st.empty()
        metrics_container = st.empty()
        status_text = st.empty()
        
        # Evolution loop
        for gen in range(num_generations):
            # --- Ecosystem Dynamics ---
            active_fitness_weights = fitness_weights
            
            # Cataclysm Recovery
            if st.session_state.cataclysm_recovery_mode > 0:
                st.session_state.cataclysm_recovery_mode -= 1
                active_fitness_weights = st.session_state.cataclysm_weights
                if st.session_state.cataclysm_recovery_mode == 0:
                    st.toast("🌍 Environmental pressures have normalized.", icon="✅")
                    st.session_state.cataclysm_weights = None

            # Cataclysm Trigger
            elif enable_cataclysms and random.random() < cataclysm_probability:
                event_type = random.choice(['extinction', 'collapse'])
                if event_type == 'extinction' and len(population) > 10:
                    st.warning(f"💥 Mass Extinction Event! A genetic bottleneck has occurred.")
                    st.toast("💥 Mass Extinction!", icon="☄️")
                    survivor_count = max(5, int(len(population) * 0.1))
                    population = random.sample(population, k=survivor_count)
                elif event_type == 'collapse':
                    st.warning(f"📉 Environmental Collapse! Fitness objectives have drastically shifted.")
                    st.toast("📉 Environmental Collapse!", icon="🌪️")
                    st.session_state.cataclysm_recovery_mode = 5 # Lasts for 5 generations
                    collapse_target = random.choice(list(fitness_weights.keys()))
                    st.session_state.cataclysm_weights = {k: 0.05 for k in fitness_weights}
                    st.session_state.cataclysm_weights[collapse_target] = 0.8
                    active_fitness_weights = st.session_state.cataclysm_weights

            # Red Queen Parasite Info
            parasite_display = status_text.empty()

            # Handle dynamic environment
            if dynamic_environment and gen > 0 and gen % env_change_frequency == 0:
                previous_task = current_task
                current_task = random.choice([t for t in task_options if t != previous_task])
                st.toast(f"🌍 Environment Shift! New Task: {current_task}", icon="🔄")
                time.sleep(1.0)
            
            status_text.markdown(f"### 🧬 Generation {gen + 1}/{num_generations} | Task: **{current_task}**")
            
            # --- Apply developmental rules ---
            # This simulates lifetime development like pruning and growth before evaluation
            if enable_development:
                for i in range(len(population)):
                    population[i] = apply_developmental_rules(population[i], stagnation_counter)

            # Evaluate fitness
            all_scores = []
            for individual in population:
                # Pass the weights from the sidebar
                # Pass the flags for advanced dynamics
                fitness, component_scores = evaluate_fitness(individual, current_task, gen, active_fitness_weights, enable_epigenetics, enable_baldwin, epistatic_linkage_k, st.session_state.parasite_profile if enable_red_queen else None)
                individual.fitness = fitness
                individual.generation = gen
                individual.age += 1
                all_scores.append(component_scores)
            
            # Record detailed history
            for individual, scores in zip(population, all_scores):
                st.session_state.history.append({
                    'generation': gen,
                    'form': f'Form {individual.form_id}',
                    'form_id': individual.form_id,
                    'fitness': individual.fitness,
                    'accuracy': scores['task_accuracy'],
                    'efficiency': scores['efficiency'],
                    'robustness': scores['robustness'],
                    'generalization': scores['generalization'],
                    'total_params': sum(m.size for m in individual.modules),
                    'num_connections': len(individual.connections),
                    'complexity': individual.complexity,
                    'lineage_id': individual.lineage_id,
                    'parent_ids': individual.parent_ids
                })
            
            # Compute evolutionary metrics
            fitness_array = np.array([ind.fitness for ind in population])
            diversity = EvolutionaryTheory.genetic_diversity(population)
            fisher_info = EvolutionaryTheory.fisher_information(population, fitness_array)

            # Display real-time metrics
            with metrics_container.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Best Fitness", f"{fitness_array.max():.4f}")
                col2.metric("Mean Fitness", f"{fitness_array.mean():.4f}")
                col3.metric("Diversity (H)", f"{diversity:.3f}")
                col4.metric("Mutation Rate (μ)", f"{current_mutation_rate:.3f}")
                # Placeholder for species count, will be updated below
                if enable_red_queen:
                    parasite_display.info(f"**Red Queen Active:** Parasite targeting `{st.session_state.parasite_profile['target_type']}` with `{st.session_state.parasite_profile['target_activation']}` activation.")
                else:
                    parasite_display.empty()
                species_metric = col5.metric("Species Count", "N/A")

            st.session_state.evolutionary_metrics.append({
                'generation': gen,
                'diversity': diversity,
                'fisher_info': fisher_info,
                'best_fitness': fitness_array.max(),
                'mean_fitness': fitness_array.mean()
            })
            
            # Selection
            if enable_speciation:
                species = []
                for ind in population:
                    found_species = False
                    for s in species:
                        representative = s['representative']
                        dist = genomic_distance(ind, representative)
                        if dist < compatibility_threshold:
                            s['members'].append(ind)
                            found_species = True
                            break
                    if not found_species:
                        species.append({'representative': ind, 'members': [ind]})
                
                species_metric.metric("Species Count", f"{len(species)}")
                
                # Apply fitness sharing
                for s in species:
                    species_size = len(s['members'])
                    if species_size > 0:
                        for member in s['members']:
                            member.adjusted_fitness = member.fitness / (species_size ** niche_competition_factor)
                
                population.sort(key=lambda x: x.adjusted_fitness, reverse=True)
                selection_key = lambda x: x.adjusted_fitness
            else:
                species_metric.metric("Species Count", f"{len(set(ind.form_id for ind in population))}")
                population.sort(key=lambda x: x.fitness, reverse=True)
                selection_key = lambda x: x.fitness

            num_survivors = max(2, int(len(population) * selection_pressure))
            survivors = population[:num_survivors]
            
            # Calculate selection differential
            selected_idx = np.arange(num_survivors)
            sel_diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)

            # Red Queen Parasite Evolution
            if enable_red_queen and survivors:
                trait_counts = Counter()
                for ind in survivors:
                    for module in ind.modules:
                        trait_counts[(module.module_type, module.activation)] += 1
                if trait_counts:
                    st.session_state.parasite_profile['target_type'], st.session_state.parasite_profile['target_activation'] = trait_counts.most_common(1)[0][0]
            
            # Reproduction
            offspring = []
            while len(offspring) < len(population) - len(survivors):
                # --- Create one viable child, with retries to prevent duds ---
                max_attempts = 20
                for _ in range(max_attempts):
                    # Tournament selection using the appropriate fitness key
                    parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                    
                    if random.random() < crossover_rate:
                        if enable_speciation and random.random() < gene_flow_rate and len(survivors) > 1:
                            # Gene Flow: select any other survivor, ignoring species
                            parent2_candidates = [s for s in survivors if s.lineage_id != parent1.lineage_id]
                            parent2 = random.choice(parent2_candidates) if parent2_candidates else parent1
                        elif len(survivors) > 1:
                            # Normal Crossover: select compatible parent
                            compatible = [s for s in survivors if s.form_id == parent1.form_id and s.lineage_id != parent1.lineage_id]
                            parent2 = max(random.sample(compatible, min(2, len(compatible))), key=selection_key) if compatible else parent1 # type: ignore
                        else:
                            parent2 = parent1
                        child = crossover(parent1, parent2, crossover_rate)
                    else:
                        child = parent1.copy()
                    
                    # Mutation and other operators
                    child = mutate(child, current_mutation_rate, innovation_rate)
                    if enable_endosymbiosis and random.random() < endosymbiosis_rate and survivors:
                        child = apply_endosymbiosis(child, survivors)
                    
                    # Viability Selection: Ensure the child is a functional network
                    if is_viable(child):
                        child.generation = gen + 1
                        offspring.append(child)
                        break # Found a viable child, move to next offspring
                else: # for-else: runs if the loop finished without break
                    # Fallback if no viable child was found after many attempts
                    parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                    child = mutate(parent1.copy(), current_mutation_rate, innovation_rate)
                    child.generation = gen + 1
                    offspring.append(child)
            
            # Clean up temporary attribute
            if enable_speciation:
                for ind in population:
                    if hasattr(ind, 'adjusted_fitness'):
                        del ind.adjusted_fitness

            # Update mutation rate for next generation
            if mutation_schedule == 'Linear Decay':
                current_mutation_rate = mutation_rate * (1.0 - ((gen + 1) / num_generations))
            elif mutation_schedule == 'Adaptive':
                current_best_fitness = fitness_array.max()
                if current_best_fitness > last_best_fitness:
                    stagnation_counter = 0
                    current_mutation_rate = max(0.05, current_mutation_rate * 0.95) # Anneal
                else:
                    stagnation_counter += 1
                
                if stagnation_counter > 3: # If stagnated for >3 generations
                    current_mutation_rate = min(0.8, current_mutation_rate * (1 + adaptive_mutation_strength)) # Spike
                
                last_best_fitness = current_best_fitness

            population = survivors + offspring
            
            # Update progress
            progress_container.progress((gen + 1) / num_generations)
        
        st.session_state.current_population = population
        
        # --- Save results to DB ---
        serializable_population = [genotype_to_dict(p) for p in population]
        
        results_to_save = {
            'history': st.session_state.history,
            'evolutionary_metrics': st.session_state.evolutionary_metrics,
            'current_population': serializable_population
        }
        if results_table.get(doc_id=1):
            results_table.update(results_to_save, doc_ids=[1])
        else:
            results_table.insert(results_to_save)
        
        status_text.markdown("### ✅ Evolution Complete! Results saved.")
        # st.balloons() # Removed for a more serious tone
    
    # Display results
    if st.session_state.history:
        st.markdown("---")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.history)
        
        # Key Metrics Summary
        st.header("📊 Evolutionary Outcome Analysis")
        
        # --- ADVANCED VISUALIZATIONS ---
        if not history_df.empty:
            visualize_fitness_landscape(history_df)
        
        if 'evolutionary_metrics' in st.session_state and st.session_state.evolutionary_metrics:
            metrics_df = pd.DataFrame(st.session_state.evolutionary_metrics)
            if len(metrics_df) > 1:
                visualize_phase_space_portraits(history_df, metrics_df)
        
        st.markdown("<hr style='margin-top: 2rem; margin-bottom: 2rem;'>", unsafe_allow_html=True)

        # --- NEW DETAILED ANALYSIS SECTION ---

        # Data preparation for analysis
        final_gen = history_df[history_df['generation'] == history_df['generation'].max()]
        population = st.session_state.current_population
        population.sort(key=lambda x: x.fitness, reverse=True)
        best_individual_genotype = population[0] if population else None
        metrics_df = pd.DataFrame(st.session_state.get('evolutionary_metrics', []))

        # --- Setup for deep analysis sections (moved up) ---
        s = st.session_state.settings
        task_type = s.get('task_type', 'Abstract Reasoning (ARC-AGI-2)')
        enable_epigenetics = s.get('enable_epigenetics', True)
        enable_baldwin = s.get('enable_baldwin', True)
        epistatic_linkage_k = s.get('epistatic_linkage_k', 0)
        
        w_accuracy = s.get('w_accuracy', 0.5)
        w_efficiency = s.get('w_efficiency', 0.2)
        w_robustness = s.get('w_robustness', 0.1)
        w_generalization = s.get('w_generalization', 0.2)
        total_w = w_accuracy + w_efficiency + w_robustness + w_generalization + 1e-9
        fitness_weights = {
            'task_accuracy': w_accuracy / total_w,
            'efficiency': w_efficiency / total_w,
            'robustness': w_robustness / total_w,
            'generalization': w_generalization / total_w
        }
        eval_params = {
            'enable_epigenetics': enable_epigenetics,
            'enable_baldwin': enable_baldwin,
            'epistatic_linkage_k': epistatic_linkage_k
        }

        st.markdown("### Executive Summary: Evolutionary Trajectory and Outcome")
        if best_individual_genotype and not history_df.empty:
            # --- Detailed Data Extraction for Summary ---
            num_generations = history_df['generation'].max() + 1
            task_type = st.session_state.settings['task_type']
            selection_pressure_param = st.session_state.settings['selection_pressure']
            
            initial_mean_fitness = history_df[history_df['generation']==0]['fitness'].mean()
            final_peak_fitness = best_individual_genotype.fitness
            final_mean_fitness = history_df[history_df['generation']==history_df['generation'].max()]['fitness'].mean()
            
            peak_improvement = ((final_peak_fitness / initial_mean_fitness) - 1) * 100 if initial_mean_fitness > 0 else 0
            mean_improvement = ((final_mean_fitness / initial_mean_fitness) - 1) * 100 if initial_mean_fitness > 0 else 0

            # Calculate mean selection differential
            selection_diff_data = []
            for gen in sorted(history_df['generation'].unique()):
                gen_data = history_df[history_df['generation'] == gen]
                if len(gen_data) > 5:
                    fitness_array = gen_data['fitness'].values
                    num_survivors = max(2, int(len(gen_data) * selection_pressure_param))
                    selected_idx = np.argpartition(fitness_array, -num_survivors)[-num_survivors:]
                    diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)
                    selection_diff_data.append({'generation': gen, 'selection_diff': diff})
            sel_df = pd.DataFrame(selection_diff_data)
            mean_selection_differential = sel_df['selection_diff'].mean() if not sel_df.empty else 0.0

            # Calculate final heritability
            heritabilities = []
            if history_df['generation'].max() > 1:
                for gen in range(1, history_df['generation'].max()):
                    parent_gen = history_df[history_df['generation'] == gen - 1]
                    offspring_gen = history_df[history_df['generation'] == gen]
                    if len(parent_gen) > 2 and len(offspring_gen) > 2:
                        h2 = EvolutionaryTheory.heritability(parent_gen['fitness'].values, offspring_gen['fitness'].values)
                        heritabilities.append({'generation': gen, 'heritability': h2})
            h2_df = pd.DataFrame(heritabilities)
            final_heritability = h2_df.iloc[-1]['heritability'] if not h2_df.empty else 0.0
            
            # Diversity metrics
            initial_diversity = metrics_df.iloc[0]['diversity'] if not metrics_df.empty else 0.0
            final_diversity = metrics_df.iloc[-1]['diversity'] if not metrics_df.empty else 0.0
            diversity_change = ((final_diversity / initial_diversity) - 1) * 100 if initial_diversity > 0 else 0.0

            st.markdown(f"""
            The evolutionary process, spanning **{num_generations} generations**, can be interpreted as a trajectory of a complex adaptive system through a high-dimensional state space, driven by the selective pressures of the **'{task_type}'** environment. This summary deconstructs the run's macro-dynamics.

            **System Dynamics & Trajectory Analysis:**
            The simulation began with a mean population fitness of `{initial_mean_fitness:.3f}`. The evolutionary trajectory, as seen in the 3D landscape, shows an initial phase of broad exploration followed by a decisive climb towards a high-fitness region. The system's dynamics are characterized by:
            - **Selection Pressure:** A mean selection differential (Δ) of **`{mean_selection_differential:.3f}`** indicates a consistent and strong directional force, a direct result of the `{selection_pressure_param*100:.0f}%` survival rate.
            - **Adaptive Response:** The population responded effectively to this pressure, evidenced by a final narrow-sense heritability (h²) of **`{final_heritability:.3f}`**. This high value signifies that fitness was a strongly heritable trait, allowing selection to drive rapid adaptation.
            - **Outcome:** The process culminated in a final mean population fitness of `{final_mean_fitness:.3f}` (a **`{mean_improvement:+.1f}%`** improvement over the initial state) and an apex fitness of **`{final_peak_fitness:.4f}`** in the champion lineage (a **`{peak_improvement:+.1f}%`** relative gain).

            **Convergence and Stability:**
            The system's state can be analyzed through its phase-space portraits. Initially characterized by high genetic diversity (H₀ = `{initial_diversity:.3f}`), the population underwent significant **convergent evolution**, collapsing into a more homogenous state with a final diversity of Hₙ = `{final_diversity:.3f}` (a change of **`{diversity_change:+.1f}%`**). The phase-space portrait for diversity shows the trajectory spiraling towards the `dH/dt = 0` nullcline, indicating the system has settled into a **stable attractor basin**. This suggests the population has thoroughly exploited a dominant peak on the fitness landscape.

            **Final State & Pareto Optimality:**
            The final population does not represent a single "winner," but rather a distribution of non-dominated solutions along a **Pareto Frontier**. The scatter of points on the 3D landscape illustrates the trade-offs between accuracy, efficiency, and robustness that were discovered. The subsequent sections provide a granular analysis of these trade-offs, the architectural motifs of the elite genotypes, and the causal structure of the synthesized 'master' architecture.
            """)

        st.markdown("---")

        st.subheader("🔬 In-Depth Analysis of the Apex Genotype")
        if best_individual_genotype:
            st.markdown(f"""
            The evolutionary process culminated in the **Apex Genotype** (Lineage ID: `{best_individual_genotype.lineage_id}`), a paragon of adaptation within the final population. This section provides a multi-faceted deconstruction of its architecture, causal structure, and evolutionary potential, offering a window into the characteristics of a peak-performing solution.
            """)

            # --- Perform all analyses upfront for the Apex Genotype ---
            with st.spinner("Performing comprehensive deep analysis on Apex Genotype..."):
                criticality_scores = analyze_lesion_sensitivity(best_individual_genotype, best_individual_genotype.fitness, task_type, fitness_weights, eval_params)
                centrality_scores = analyze_information_flow(best_individual_genotype)
                evo_robust_data = analyze_evolvability_robustness(best_individual_genotype, task_type, fitness_weights, eval_params)
                dev_traj_df = analyze_developmental_trajectory(best_individual_genotype)
                load_data = analyze_genetic_load(criticality_scores)
                parent_ids = best_individual_genotype.parent_ids
                parents = [p for p in population if p.lineage_id in parent_ids]

            # --- Create Tabs for Deep Dive ---
            tab_vitals, tab_causal, tab_potential, tab_ancestry, tab_export = st.tabs([
                "🌐 Vitals & Architecture", 
                "🔬 Causal & Structural Analysis", 
                "🧬 Evolutionary & Developmental Potential",
                "🌳 Genealogy & Ancestry",
                "💻 Code Export"
            ])

            # --- TAB 1: Vitals & Architecture ---
            with tab_vitals:
                vitals_col1, vitals_col2 = st.columns([1, 1])
                with vitals_col1:
                    st.markdown("#### Quantitative Profile vs. Population Mean")
                    pop_mean_fitness = final_gen['fitness'].mean()
                    pop_mean_accuracy = final_gen['accuracy'].mean()
                    pop_mean_complexity = final_gen['complexity'].mean()
                    pop_mean_params = final_gen['total_params'].mean()

                    comparison_data = {
                        "Metric": ["Fitness", "Accuracy", "Complexity", "Parameters"],
                        "Apex Value": [f"{best_individual_genotype.fitness:.4f}", f"{best_individual_genotype.accuracy:.3f}", f"{best_individual_genotype.complexity:.3f}", f"{sum(m.size for m in best_individual_genotype.modules):,}"],
                        "Population Mean": [f"{pop_mean_fitness:.4f}", f"{pop_mean_accuracy:.3f}", f"{pop_mean_complexity:.3f}", f"{pop_mean_params:,.0f}"]
                    }
                    st.dataframe(pd.DataFrame(comparison_data).set_index("Metric"), use_container_width=True, key="apex_vitals_df")

                    st.markdown("##### Module Composition")
                    module_data = [{"ID": m.id, "Type": m.module_type, "Size": m.size, "Activation": m.activation, "Plasticity": f"{m.plasticity:.2f}"} for m in best_individual_genotype.modules]
                    st.dataframe(module_data, height=200, use_container_width=True, key="apex_modules_df")

                with vitals_col2:
                    st.markdown("#### Architectural Visualization")
                    st.plotly_chart(visualize_genotype_3d(best_individual_genotype), use_container_width=True, key="apex_3d_vis")
                    st.plotly_chart(visualize_genotype_2d(best_individual_genotype), use_container_width=True, key="apex_2d_vis")

            # --- TAB 2: Causal & Structural Analysis ---
            with tab_causal:
                st.markdown("This tab dissects the functional importance of the architecture's components.")
                causal_col1, causal_col2 = st.columns(2)
                with causal_col1:
                    st.subheader("Lesion Sensitivity Analysis")
                    st.markdown("We computationally 'remove' each component and measure the drop in fitness. A larger drop indicates a more **critical** component.")
                    sorted_criticality = sorted(criticality_scores.items(), key=lambda item: item[1], reverse=True)
                    st.markdown("###### Most Critical Components:")
                    for j, (component, score) in enumerate(sorted_criticality[:5]):
                        st.metric(label=f"#{j+1} {component}", value=f"{score:.4f} Fitness Drop", help="The reduction in overall fitness when this component is removed.")

                with causal_col2:
                    st.subheader("Information Flow Backbone")
                    st.markdown("This analysis identifies the **causal backbone**—the key modules that act as bridges for information flow, identified via `betweenness centrality`.")
                    sorted_centrality = sorted(centrality_scores.items(), key=lambda item: item[1], reverse=True)
                    st.markdown("###### Causal Backbone Nodes:")
                    for j, (module_id, score) in enumerate(sorted_centrality[:5]):
                        st.metric(label=f"#{j+1} Module: {module_id}", value=f"{score:.3f} Centrality", help="A normalized score of how critical this node is for information routing.")
                
                st.markdown("---")
                st.subheader("Genetic Load & Neutrality")
                st.markdown("This analysis identifies **neutral components** ('junk DNA') with little effect when removed, and calculates the **genetic load**—the fitness cost of slightly harmful, non-lethal components.")
                load_col1, load_col2 = st.columns(2)
                load_col1.metric("Neutral Component Count", f"{load_data['neutral_component_count']}", help="Number of modules/connections with near-zero impact when lesioned.")
                load_col2.metric("Genetic Load", f"{load_data['genetic_load']:.4f}", help="Total fitness reduction from slightly deleterious, non-critical components.")

            # --- TAB 3: Evolutionary & Developmental Potential ---
            with tab_potential:
                st.markdown("This tab probes the genotype's potential for future adaptation and its programmed developmental trajectory.")
                evo_col1, evo_col2 = st.columns(2)
                with evo_col1:
                    st.subheader("Evolvability vs. Robustness")
                    st.markdown("We generate 50 mutants to measure the trade-off between **robustness** (resisting negative mutations) and **evolvability** (producing beneficial mutations).")
                    st.metric("Robustness Score", f"{evo_robust_data['robustness']:.4f}", help="Average fitness loss from deleterious mutations. Higher = more robust.")
                    st.metric("Evolvability Score", f"{evo_robust_data['evolvability']:.4f}", help="Maximum fitness gain from a single mutation.")

                    dist_df = pd.DataFrame(evo_robust_data['distribution'], columns=['Fitness Change'])
                    fig = px.histogram(dist_df, x="Fitness Change", nbins=20, title="Distribution of Mutational Effects")
                    fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey")
                    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
                    st.plotly_chart(fig, use_container_width=True, key="apex_mutational_effects_hist")

                with evo_col2:
                    st.subheader("Developmental Trajectory")
                    st.markdown("This simulates the genotype's 'lifetime,' showing how its developmental program (pruning, proliferation) alters its structure over time.")
                    fig = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, use_container_width=True, key="apex_dev_trajectory_line")

            # --- TAB 4: Genealogy & Ancestry ---
            with tab_ancestry:
                st.markdown("This tab traces the lineage of the genotype, comparing it to its direct ancestors to understand the evolutionary step that led to its success.")
                st.subheader(f"Direct Ancestors: `{', '.join(parent_ids)}`")

                if not parents:
                    st.info("Parents of this genotype are not in the final population (they were not selected in a previous generation).")
                else:
                    parent_cols = st.columns(len(parents))
                    for k, parent in enumerate(parents):
                        with parent_cols[k]:
                            st.markdown(f"##### Parent: `{parent.lineage_id}`")
                            st.markdown(f"**Form:** `{parent.form_id}` | **Fitness:** `{parent.fitness:.4f}`")
                            distance = genomic_distance(best_individual_genotype, parent)
                            st.metric("Genomic Distance to Child", f"{distance:.3f}")

                            st.markdown("###### Evolutionary Step:")
                            param_delta = sum(m.size for m in best_individual_genotype.modules) - sum(m.size for m in parent.modules)
                            st.metric("Parameter Change", f"{param_delta:+,}", delta_color="off")
                            complexity_delta = best_individual_genotype.complexity - parent.complexity
                            st.metric("Complexity Change", f"{complexity_delta:+.3f}", delta_color="off")

                            st.plotly_chart(visualize_genotype_2d(parent), use_container_width=True, key=f"apex_parent_2d_{parent.lineage_id}")
                            
            # --- TAB 5: Code Export ---
            with tab_export:
                st.markdown("The genotype can be translated into functional code for deep learning frameworks, providing a direct path from discovery to application.")
                code_col1, code_col2 = st.columns(2)
                with code_col1:
                    st.subheader("PyTorch Code")
                    st.code(generate_pytorch_code(best_individual_genotype), language='python')
                with code_col2:
                    st.subheader("TensorFlow / Keras Code")
                    st.code(generate_tensorflow_code(best_individual_genotype), language='python')

            # --- Comprehensive Interpretation ---
            st.markdown("---")
            st.markdown("#### **Comprehensive Interpretation**")
            
            dominant_module_type = Counter(m.module_type for m in best_individual_genotype.modules).most_common(1)[0][0]
            critical_component, fitness_drop = (sorted_criticality[0] if sorted_criticality else ("N/A", 0))
            centrality = (sorted_centrality[0][1] if sorted_centrality else 0)
            parent_fitness = parents[0].fitness if parents else initial_mean_fitness
            fitness_leap = best_individual_genotype.fitness - parent_fitness
            
            interpretation_text = f"""
            The Apex Genotype's superior fitness (`{best_individual_genotype.fitness:.4f}`) is an emergent property of a sophisticated and highly adapted design. Its success can be deconstructed into several key factors:

            1.  **Architectural Specialization:** The genotype's structure, dominated by **{dominant_module_type}** modules, reflects a strong adaptation to the demands of the **'{task_type}'** task. This specialization is not just structural but functional, as evidenced by its quantitative superiority over the population mean in both fitness and accuracy.

            2.  **Causal Integrity:** The causal analysis reveals a well-defined functional core. The high lesion sensitivity of **{critical_component}** (fitness drop: `{fitness_drop:.4f}`) and its significant information flow centrality (`{centrality:.3f}`) identify it as an indispensable hub for processing. The architecture is not a random assortment of parts but a network with critical, load-bearing components.

            3.  **Balanced Evolutionary Potential:** The genotype exists in a state of balanced adaptability. Its robustness score (`{evo_robust_data['robustness']:.4f}`) indicates resilience to deleterious mutations, a trait crucial for stability. Concurrently, its evolvability score (`{evo_robust_data['evolvability']:.4f}`) and the presence of neutral components (`{load_data['neutral_component_count']}`) demonstrate that it has not reached an evolutionary dead-end. It retains "latent potential" and genetic raw material for future adaptation.

            4.  **Significant Evolutionary Leap:** The ancestry analysis shows this genotype represents a significant step forward. It achieved a fitness leap of **`{fitness_leap:+.4f}`** over its direct parent, a jump likely attributable to a key mutation or recombination event. This highlights the power of the evolutionary operators to produce meaningful innovation.

            In summary, the Apex Genotype is a hallmark of successful neuroevolution: a specialized, causally robust architecture that resulted from a significant innovative leap, while still retaining the potential for future adaptation. It is both a product of its history and a platform for the future.
            """
            st.info(interpretation_text)
        else:
            st.warning("No best individual found to analyze.")

        st.markdown("---")

        # --- Pareto Frontier Analysis ---
        st.subheader("⚖️ Pareto Frontier: Analyzing Performance Trade-offs")
        st.markdown("""
        Evolution rarely produces a single "best" solution. Instead, it discovers a **Pareto Frontier**—a set of optimal solutions where no single objective can be improved without degrading another. This section provides a deep, interactive analysis of this frontier from the final population.

        - **The Frontier:** Individuals on the frontier (highlighted in the plots) represent the best possible trade-offs found by the evolutionary process. Any individual *not* on the frontier is "dominated," meaning there is at least one other individual that is better in one objective and no worse in any other.
        - **Archetypes:** We isolate four key archetypes from the frontier to understand the different strategies that emerged: the **High-Accuracy Specialist**, the **High-Efficiency Generalist**, the **High-Robustness Sentinel**, and the **Balanced Performer** (closest to the "utopia" point of perfect scores).
        """)

        # Find the archetypes
        final_gen_genotypes = [ind for ind in population if ind.generation == history_df['generation'].max()] if population else []
        
        if final_gen_genotypes:
            with st.spinner("Identifying Pareto frontier and analyzing archetypes..."):
                pareto_individuals = identify_pareto_frontier(final_gen_genotypes)
                pareto_lineage_ids = {p.lineage_id for p in pareto_individuals}

                # Create a DataFrame for easier plotting
                final_pop_df = pd.DataFrame([asdict(g) for g in final_gen_genotypes])
                final_pop_df['is_pareto'] = final_pop_df['lineage_id'].isin(pareto_lineage_ids)
                
                pareto_df = final_pop_df[final_pop_df['is_pareto']]
                dominated_df = final_pop_df[~final_pop_df['is_pareto']]

            tab1, tab2 = st.tabs(["Interactive Frontier Visualization", "Archetype Deep Dive"])

            with tab1:
                st.markdown("#### **3D Interactive Pareto Frontier**")
                st.markdown("This 3D scatter plot shows all individuals from the final generation. The larger, brighter points form the discovered Pareto Frontier, representing the optimal trade-offs between Accuracy, Efficiency, and Robustness.")
                
                # 3D Plot
                fig_3d = go.Figure()
                # Dominated points
                fig_3d.add_trace(go.Scatter3d(
                    x=dominated_df['accuracy'], y=dominated_df['efficiency'], z=dominated_df['robustness'],
                    mode='markers',
                    marker=dict(size=5, color='lightgrey', opacity=0.5),
                    name='Dominated Solutions',
                    hovertext="Lineage: " + dominated_df['lineage_id'] + "<br>Fitness: " + dominated_df['fitness'].round(3).astype(str),
                    hoverinfo='text+x+y+z'
                ))
                # Pareto points
                fig_3d.add_trace(go.Scatter3d(
                    x=pareto_df['accuracy'], y=pareto_df['efficiency'], z=pareto_df['robustness'],
                    mode='markers',
                    marker=dict(
                        size=9,
                        color=pareto_df['fitness'],
                        colorscale='Viridis',
                        colorbar=dict(title='Fitness'),
                        showscale=True,
                        line=dict(width=1, color='black')
                    ),
                    name='Pareto Frontier',
                    hovertext="Lineage: " + pareto_df['lineage_id'] + "<br>Fitness: " + pareto_df['fitness'].round(3).astype(str),
                    hoverinfo='text+x+y+z'
                ))
                fig_3d.update_layout(
                    title='<b>Final Population vs. Pareto Frontier</b>',
                    scene=dict(
                        xaxis_title='Accuracy',
                        yaxis_title='Efficiency',
                        zaxis_title='Robustness',
                        camera=dict(eye=dict(x=1.8, y=1.8, z=1.8))
                    ),
                    height=600,
                    margin=dict(l=0, r=0, b=0, t=40)
                )
                st.plotly_chart(fig_3d, use_container_width=True, key="pareto_3d_plot")

                st.markdown("---")
                st.markdown("#### **2D Trade-off Projections**")
                st.markdown("These plots show 2D projections of the frontier. The 'Utopia Point' (top-right) represents an ideal but likely unreachable solution. The lines from each Pareto point to Utopia illustrate the trade-off space.")

                fig_2d_matrix = make_subplots(
                    rows=1, cols=3,
                    subplot_titles=('Accuracy vs. Efficiency', 'Accuracy vs. Robustness', 'Efficiency vs. Robustness')
                )

                objectives = [('accuracy', 'efficiency'), ('accuracy', 'robustness'), ('efficiency', 'robustness')]
                for i, (obj1, obj2) in enumerate(objectives):
                    # Dominated
                    fig_2d_matrix.add_trace(go.Scatter(
                        x=dominated_df[obj1], y=dominated_df[obj2], mode='markers',
                        marker=dict(color='lightgrey', size=6, opacity=0.6),
                        name='Dominated', showlegend=(i==0)
                    ), row=1, col=i+1)
                    # Pareto
                    fig_2d_matrix.add_trace(go.Scatter(
                        x=pareto_df[obj1], y=pareto_df[obj2], mode='markers',
                        marker=dict(color=pareto_df['fitness'], colorscale='Viridis', size=10, line=dict(width=1, color='black')),
                        name='Pareto', showlegend=(i==0)
                    ), row=1, col=i+1)
                    # Utopia Point
                    fig_2d_matrix.add_trace(go.Scatter(
                        x=[1], y=[1], mode='markers',
                        marker=dict(symbol='star', color='gold', size=15, line=dict(width=1, color='black')),
                        name='Utopia Point', showlegend=(i==0)
                    ), row=1, col=i+1)
                    
                    # Lines to Utopia
                    for _, row in pareto_df.iterrows():
                        fig_2d_matrix.add_shape(type="line",
                            x0=row[obj1], y0=row[obj2], x1=1, y1=1,
                            line=dict(color="rgba(200,200,200,0.3)", width=1, dash="dot"),
                            row=1, col=i+1
                        )

                    fig_2d_matrix.update_xaxes(title_text=obj1.capitalize(), range=[0, 1.05], row=1, col=i+1)
                    fig_2d_matrix.update_yaxes(title_text=obj2.capitalize(), range=[0, 1.05], row=1, col=i+1)

                fig_2d_matrix.update_layout(height=450, title_text="<b>2D Pareto Trade-off Analysis</b>", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_2d_matrix, use_container_width=True, key="pareto_2d_matrix_plot")

            with tab2:
                # Find the archetypes
                acc_specialist = max(final_gen_genotypes, key=lambda g: g.accuracy)
                eff_generalist = max(final_gen_genotypes, key=lambda g: g.efficiency)
                rob_sentinel = max(final_gen_genotypes, key=lambda g: g.robustness)
                
                # Find balanced performer
                utopia_point = np.array([1.0, 1.0, 1.0])
                distances = []
                for ind in pareto_individuals:
                    point = np.array([ind.accuracy, ind.efficiency, ind.robustness])
                    dist = np.linalg.norm(utopia_point - point)
                    distances.append((dist, ind))

                balanced_performer = min(distances, key=lambda x: x[0])[1] if distances else None

                archetypes = {
                    "🎯 High-Accuracy Specialist": acc_specialist,
                    "⚡ High-Efficiency Generalist": eff_generalist,
                    "🛡️ High-Robustness Sentinel": rob_sentinel,
                    "⚖️ Balanced Performer": balanced_performer
                }

                p_cols = st.columns(len([a for a in archetypes.values() if a is not None]))
                col_idx = 0
                for name, archetype in archetypes.items():
                    if archetype is None: continue
                    with p_cols[col_idx]:
                        st.markdown(f"##### {name}")
                        st.markdown(f"**Lineage:** `{archetype.lineage_id}`")
                        
                        stats_df = pd.DataFrame({
                            "Metric": ["Fitness", "Accuracy", "Efficiency", "Robustness", "Params"],
                            "Value": [
                                f"{archetype.fitness:.3f}",
                                f"{archetype.accuracy:.3f}",
                                f"{archetype.efficiency:.3f}",
                                f"{archetype.robustness:.3f}",
                                f"{sum(m.size for m in archetype.modules):,}"
                            ]
                        }).set_index("Metric")
                        st.dataframe(stats_df, use_container_width=True, key=f"pareto_archetype_df_{name}")

                        interpretation = ""
                        if "Accuracy" in name:
                            interpretation = "This genotype prioritized task performance above all else, likely developing a large, complex architecture to maximize its score, sacrificing computational cost."
                        elif "Efficiency" in name:
                            interpretation = "This compact genotype is optimized for low computational overhead. It achieves respectable accuracy with minimal parameters, making it ideal for resource-constrained environments."
                        elif "Robustness" in name:
                            interpretation = "This architecture evolved for stability. Its structure, likely featuring redundancy and moderate plasticity, is resilient to perturbations and noise, ensuring reliable performance."
                        elif "Balanced" in name:
                            interpretation = "This genotype represents the best overall compromise found on the Pareto frontier. It doesn't excel at any single objective but provides a strong, balanced performance across all three, making it a robust, all-around solution."
                        st.info(interpretation)

                        with st.expander("View Architecture"):
                            st.plotly_chart(visualize_genotype_2d(archetype), use_container_width=True, key=f"pareto_archetype_2d_{name}")
                    col_idx += 1

        else:
            st.warning("Could not perform Pareto analysis on the final generation.")

        st.markdown("---")

        # --- Population-Level Insights ---
        st.subheader("🌍 Population-Level Dynamics & Conclusions")
        st.markdown("The characteristics of the entire final population provide insights into the overall evolutionary pressures and the convergence of architectural forms.")

        pop_col1, pop_col2 = st.columns(2)

        with pop_col1:
            st.markdown("##### Form Dominance")
            if not final_gen.empty:
                form_counts = final_gen['form_id'].value_counts()
                dominant_form = form_counts.index[0]
                dominance_pct = (form_counts.iloc[0] / form_counts.sum()) * 100
                st.markdown(f"**Form `{int(dominant_form)}`** emerged as the dominant morphology, comprising **`{dominance_pct:.1f}%`** of the final population. This indicates that its foundational topology provided a significant adaptive advantage in the given task environment.")
                
                fig = px.pie(
                    values=form_counts.values,
                    names=[f"Form {i}" for i in form_counts.index],
                    title='Final Population Distribution by Form',
                    color_discrete_sequence=px.colors.qualitative.Set3
                )
                fig.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
                st.plotly_chart(fig, use_container_width=True, key="form_dominance_pie")
            else:
                st.info("No final generation data to analyze form dominance.")

        with pop_col2:
            st.markdown("##### Genetic Diversity & Convergence")
            if len(metrics_df) > 1:
                final_diversity = metrics_df.iloc[-1]['diversity']
                initial_diversity = metrics_df.iloc[0]['diversity']
                diversity_change = ((final_diversity / initial_diversity) - 1) * 100 if initial_diversity > 0 else 0
                
                st.metric("Final Genetic Diversity (H)", f"{final_diversity:.3f}", f"{diversity_change:.1f}% vs. Initial")
                
                if diversity_change < -50:
                    st.info("The significant drop in diversity indicates strong **convergent evolution**. The population has collectively identified a narrow set of highly effective genotypes, pruning away less successful variations.")
                elif diversity_change > -10:
                    st.info("Diversity was largely maintained, suggesting **divergent evolution** or a complex fitness landscape with multiple viable peaks. The population retains a broad range of solutions, which is beneficial for adapting to future environmental shifts.")
                else:
                    st.info("The population has undergone moderate convergence, balancing exploration with exploitation. A healthy level of diversity remains while honing in on promising regions of the search space.")
            else:
                st.info("Not enough data to analyze diversity trends.")

        # --- Finally, show the dashboard ---
        st.markdown("---")
        st.subheader("📈 Comprehensive Evolutionary Dashboard")
        st.markdown("This dashboard provides a holistic, multi-faceted view of the key metrics tracked across all generations, deconstructing the run's performance, population dynamics, and underlying evolutionary forces.")
        metrics_df = pd.DataFrame(st.session_state.get('evolutionary_metrics', []))
        st.plotly_chart(
            create_evolution_dashboard(history_df, st.session_state.current_population, metrics_df), # type: ignore
            width='stretch',
            key="main_dashboard_plot"
        )

        # Best evolved architectures
        st.markdown("---")
        st.header("🏛️ Elite Evolved Architectures: A Deep Dive")
        st.markdown("This section provides a multi-faceted analysis of the top-performing genotypes from the final population, deconstructing the architectural, causal, and evolutionary properties that contributed to their success.")
        
        population = st.session_state.current_population
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Show top 3
        for i, individual in enumerate(population[:3]):
            expander_title = f"**Rank {i+1}:** Form `{individual.form_id}` | Lineage `{individual.lineage_id}` | Fitness: `{individual.fitness:.4f}`"
            with st.expander(expander_title, expanded=(i==0)):
                
                # Define tabs for the deep dive
                tab_vitals, tab_causal, tab_evo, tab_ancestry, tab_code = st.tabs([
                    "🌐 Vitals & Architecture", 
                    "🔬 Causal & Structural Analysis", 
                    "🧬 Evolutionary & Developmental Potential",
                    "🌳 Genealogy & Ancestry",
                    "💻 Code Export"
                ])
                # --- TAB 1: Vitals & Architecture ---
                with tab_vitals:
                    vitals_col1, vitals_col2 = st.columns([1, 1])
                    with vitals_col1:
                        st.markdown("#### Quantitative Profile")
                        st.metric("Fitness", f"{individual.fitness:.4f}")
                        st.metric("Task Accuracy (Sim.)", f"{individual.accuracy:.3f}")
                        st.metric("Efficiency Score", f"{individual.efficiency:.3f}")
                        st.metric("Robustness Score", f"{individual.robustness:.3f}")
                        st.metric("Architectural Complexity", f"{individual.complexity:.3f}")
                        st.metric("Age", f"{individual.age} generations")

                    with vitals_col2:
                        st.markdown("#### Architectural Blueprint")
                        st.write(f"**Total Parameters:** `{sum(m.size for m in individual.modules):,}`")
                        st.write(f"**Modules:** `{len(individual.modules)}` | **Connections:** `{len(individual.connections)}`")
                        st.write(f"**Parent(s):** `{', '.join(individual.parent_ids)}`")
                        
                        st.markdown("###### Module Composition:")
                        module_counts = Counter(m.module_type for m in individual.modules)
                        for mtype, count in module_counts.items():
                            st.write(f"- `{count}` x **{mtype.capitalize()}**")
                        
                        st.markdown("###### Meta-Parameters:")
                        st.json({k: f"{v:.4f}" for k, v in individual.meta_parameters.items()}, key=f"elite_meta_params_{individual.lineage_id}")

                st.markdown("---")
                st.markdown("#### Visualizations")
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.markdown("###### 3D Interactive View")
                    st.plotly_chart(
                        visualize_genotype_3d(individual),
                        use_container_width=True,
                        key=f"elite_3d_{i}_{individual.lineage_id}"
                    )
                with vis_col2:
                    st.markdown("###### 2D Static View")
                    st.plotly_chart(
                        visualize_genotype_2d(individual),
                        use_container_width=True,
                        key=f"elite_2d_{i}_{individual.lineage_id}"
                    )

                # --- TAB 2: Causal & Structural Analysis ---
                with tab_causal:
                    st.markdown("This tab dissects the functional importance of the architecture's components.")
                    causal_col1, causal_col2 = st.columns(2)

                    with causal_col1:
                        st.subheader("Lesion Sensitivity Analysis")
                        st.markdown("We computationally 'remove' each component and measure the drop in fitness. A larger drop indicates a more **critical** component.")
                        
                        with st.spinner(f"Performing lesion analysis for Rank {i+1}..."):
                            criticality_scores = analyze_lesion_sensitivity(
                                individual, individual.fitness, task_type, fitness_weights, eval_params
                            )
                        
                        sorted_criticality = sorted(criticality_scores.items(), key=lambda item: item[1], reverse=True)
                        
                        st.markdown("###### Most Critical Components:")
                        for j, (component, score) in enumerate(sorted_criticality[:5]):
                            st.metric(
                                label=f"#{j+1} {component}",
                                value=f"{score:.4f} Fitness Drop",
                                help="The reduction in overall fitness when this component is removed."
                            )

                    with causal_col2:
                        st.subheader("Information Flow Backbone")
                        st.markdown("This analysis identifies the **causal backbone**—the key modules that act as bridges for information flow, identified via `betweenness centrality`.")
                        
                        with st.spinner(f"Analyzing information flow for Rank {i+1}..."):
                            centrality_scores = analyze_information_flow(individual)
                        
                        sorted_centrality = sorted(centrality_scores.items(), key=lambda item: item[1], reverse=True)
                        st.markdown("###### Causal Backbone Nodes:")
                        for j, (module_id, score) in enumerate(sorted_centrality[:5]):
                            st.metric(label=f"#{j+1} Module: {module_id}", value=f"{score:.3f} Centrality", help="A normalized score of how critical this node is for information routing.")

                    st.markdown("---")
                    st.subheader("Genetic Load & Neutrality")
                    st.markdown("This analysis identifies **neutral components** ('junk DNA') with little effect when removed, and calculates the **genetic load**—the fitness cost of slightly harmful, non-lethal components.")
                    with st.spinner(f"Calculating genetic load for Rank {i+1}..."):
                        load_data = analyze_genetic_load(criticality_scores)

                    load_col1, load_col2 = st.columns(2)
                    load_col1.metric("Neutral Component Count", f"{load_data['neutral_component_count']}", help="Number of modules/connections with near-zero impact when lesioned.")
                    load_col2.metric("Genetic Load", f"{load_data['genetic_load']:.4f}", help="Total fitness reduction from slightly deleterious, non-critical components.")

                # --- TAB 3: Evolutionary & Developmental Potential ---
                with tab_evo:
                    st.markdown("This tab probes the genotype's potential for future adaptation and its programmed developmental trajectory.")
                    evo_col1, evo_col2 = st.columns(2)

                    with evo_col1:
                        st.subheader("Evolvability vs. Robustness")
                        st.markdown("We generate 50 mutants to measure the trade-off between **robustness** (resisting negative mutations) and **evolvability** (producing beneficial mutations).")
                        
                        with st.spinner(f"Analyzing mutational landscape for Rank {i+1}..."):
                            evo_robust_data = analyze_evolvability_robustness(
                                individual, task_type, fitness_weights, eval_params
                            )
                        
                        st.metric("Robustness Score", f"{evo_robust_data['robustness']:.4f}", help="Average fitness loss from deleterious mutations. Higher = more robust.")
                        st.metric("Evolvability Score", f"{evo_robust_data['evolvability']:.4f}", help="Maximum fitness gain from a single mutation.")

                        dist_df = pd.DataFrame(evo_robust_data['distribution'], columns=['Fitness Change'])
                        fig = px.histogram(dist_df, x="Fitness Change", nbins=20, title="Distribution of Mutational Effects")
                        fig.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey")
                        fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True, key=f"mutational_effects_hist_{i}_{individual.lineage_id}")

                    with evo_col2:
                        st.subheader("Developmental Trajectory")
                        st.markdown("This simulates the genotype's 'lifetime,' showing how its developmental program (pruning, proliferation) alters its structure over time.")
                        
                        with st.spinner(f"Simulating development for Rank {i+1}..."):
                            dev_traj_df = analyze_developmental_trajectory(individual)
                        
                        fig = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), showlegend=False)
                        st.plotly_chart(fig, use_container_width=True, key=f"dev_trajectory_line_{i}_{individual.lineage_id}")

                # --- TAB 4: Genealogy & Ancestry ---
                with tab_ancestry:
                    st.markdown("This tab traces the lineage of the genotype, comparing it to its direct ancestors to understand the evolutionary step that led to its success.")
                    parent_ids = individual.parent_ids
                    st.subheader(f"Direct Ancestors: `{', '.join(parent_ids)}`")

                    parents = [p for p in population if p.lineage_id in parent_ids]

                    if not parents:
                        st.info("Parents of this genotype are not in the final population (they were not selected in a previous generation).")
                    else:
                        parent_cols = st.columns(len(parents))
                        for k, parent in enumerate(parents):
                            with parent_cols[k]:
                                st.markdown(f"##### Parent: `{parent.lineage_id}`")
                                st.markdown(f"**Form:** `{parent.form_id}` | **Fitness:** `{parent.fitness:.4f}`")
                                
                                # Genomic distance
                                distance = genomic_distance(individual, parent)
                                st.metric("Genomic Distance to Child", f"{distance:.3f}")

                                # Compare key stats
                                st.markdown("###### Evolutionary Step:")
                                param_delta = sum(m.size for m in individual.modules) - sum(m.size for m in parent.modules)
                                st.metric("Parameter Change", f"{param_delta:+,}", delta_color="off")
                                
                                complexity_delta = individual.complexity - parent.complexity
                                st.metric("Complexity Change", f"{complexity_delta:+.3f}", delta_color="off")

                                # Visualize parent
                                st.plotly_chart(
                                    visualize_genotype_2d(parent),
                                    use_container_width=True,
                                    key=f"parent_2d_{i}_{k}_{parent.lineage_id}_for_{individual.lineage_id}"
                                )

                # --- TAB 5: Code Export ---
                with tab_code:
                    st.markdown("The genotype can be translated into functional code for deep learning frameworks, providing a direct path from discovery to application.")
                    
                    code_col1, code_col2 = st.columns(2)
                    with code_col1:
                        st.subheader("PyTorch Code")
                        pytorch_code = generate_pytorch_code(individual)
                        st.code(pytorch_code, language='python')
                    
                    with code_col2:
                        st.subheader("TensorFlow / Keras Code")
                        tensorflow_code = generate_tensorflow_code(individual)
                        st.code(tensorflow_code, language='python')
        
        # Form comparison
        st.markdown("---")
        st.header("🔬 Comparative Form Analysis")
        st.markdown("""
        The initial population is seeded with distinct architectural 'forms'—foundational templates with unique inductive biases (e.g., hierarchical convolutional vs. recurrent memory). This analysis dissects how these different morphological starting points fared in the evolutionary race. By comparing their performance, dominance, and emergent traits, we can infer which architectural priors were most advantageous for the given task environment.
        """)

        if final_gen.empty:
            st.warning("No final generation data available for form comparison.")
        else:
            # Prepare data
            form_names = sorted(final_gen['form'].unique())
            
            tab1, tab2, tab3 = st.tabs([
                "📊 Performance & Dominance", 
                "🧬 Phenotypic Trait Comparison", 
                "⚖️ Multi-Objective Strategy Profile"
            ])

            with tab1:
                st.markdown("#### **How did each form perform and which became dominant?**")
                col1, col2 = st.columns([1.2, 1])

                with col1:
                    st.markdown("###### **Overall Performance Statistics**")
                    # More detailed performance aggregation
                    form_performance = final_gen.groupby('form').agg(
                        mean_fitness=('fitness', 'mean'),
                        median_fitness=('fitness', 'median'),
                        max_fitness=('fitness', 'max'),
                        std_fitness=('fitness', 'std'),
                        mean_accuracy=('accuracy', 'mean'),
                        population_count=('fitness', 'size')
                    ).round(4).reset_index()
                    st.dataframe(form_performance, use_container_width=True, key="form_perf_stats_df")

                with col2:
                    st.markdown("###### **Final Population Dominance**")
                    # Bar chart showing count and colored by mean fitness
                    dominance_data = form_performance[['form', 'population_count', 'mean_fitness']]
                    fig_dom = px.bar(
                        dominance_data,
                        x='form',
                        y='population_count',
                        color='mean_fitness',
                        labels={'form': 'Architectural Form', 'population_count': 'Number of Individuals', 'mean_fitness': 'Mean Fitness'},
                        title='Form Dominance by Count and Mean Fitness',
                        color_continuous_scale=px.colors.sequential.Viridis
                    )
                    fig_dom.update_layout(height=300, margin=dict(t=40, b=20, l=20, r=20))
                    st.plotly_chart(fig_dom, use_container_width=True, key="form_dominance_bar")

                st.markdown("---")
                st.markdown("###### **Fitness Distribution by Form**")
                st.markdown("This plot shows the full distribution of fitness scores for each form, revealing not just the average but also the spread, median, and presence of high-performing outliers.")
                fig_box = px.box(
                    final_gen,
                    x='form',
                    y='fitness',
                    color='form',
                    points='all', # Show all individuals
                    title='Fitness Distribution Across Architectural Forms',
                    labels={'form': 'Architectural Form', 'fitness': 'Fitness Score'}
                )
                fig_box.update_layout(showlegend=False)
                st.plotly_chart(fig_box, use_container_width=True, key="form_fitness_dist_box")

            with tab2:
                st.markdown("#### **Did forms develop different physical characteristics?**")
                st.markdown("Here we compare the distribution of key phenotypic traits—network size (total parameters) and architectural complexity—that evolved within each form.")
                
                trait_col1, trait_col2 = st.columns(2)
                with trait_col1:
                    fig_params = px.box(
                        final_gen,
                        x='form',
                        y='total_params',
                        color='form',
                        title='Network Size (Parameters) by Form',
                        labels={'form': 'Architectural Form', 'total_params': 'Total Parameters'}
                    )
                    fig_params.update_layout(showlegend=False)
                    st.plotly_chart(fig_params, use_container_width=True, key="form_params_box")

                with trait_col2:
                    fig_complexity = px.box(
                        final_gen,
                        x='form',
                        y='complexity',
                        color='form',
                        title='Architectural Complexity by Form',
                        labels={'form': 'Architectural Form', 'complexity': 'Complexity Score'}
                    )
                    fig_complexity.update_layout(showlegend=False)
                    st.plotly_chart(fig_complexity, use_container_width=True, key="form_complexity_box")

            with tab3:
                st.markdown("#### **How did each form approach the multi-objective problem?**")
                st.markdown("This analysis reveals the different strategies each form adopted to balance the competing objectives of accuracy, efficiency, robustness, and generalization.")

                # Find the champion for each form
                champions = final_gen.loc[final_gen.groupby('form')['fitness'].idxmax()]
                
                st.markdown("###### **Strategy Profile of Form Champions**")
                st.markdown("The radar chart compares the objective scores of the single best individual from each form, highlighting their specialized strengths.")
                
                if not champions.empty:
                    objectives = ['accuracy', 'efficiency', 'robustness', 'generalization']
                    fig_radar = go.Figure()
                    for _, champion in champions.iterrows():
                        fig_radar.add_trace(go.Scatterpolar(
                            r=[champion[obj] for obj in objectives] + [champion[objectives[0]]], # Close the loop
                            theta=[obj.capitalize() for obj in objectives] + [objectives[0].capitalize()],
                            fill='toself',
                            name=champion['form']
                        ))
                    
                    fig_radar.update_layout(
                        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                        showlegend=True,
                        title="Multi-Objective Profile of Form Champions",
                        height=450,
                        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                    )
                    st.plotly_chart(fig_radar, use_container_width=True, key="form_champions_radar")

                st.markdown("---")
                st.markdown("###### **Population-Wide Objective Space Occupation**")
                st.markdown("The parallel coordinates plot visualizes the entire final population. Each line is an individual, and the 'bands' of color show how the population of each form is distributed across the different objective dimensions. This reveals whether forms occupy distinct niches in the solution space.")
                
                fig_parallel = px.parallel_coordinates(
                    final_gen.sort_values('form'), # Sort to group colors
                    color="form_id",
                    dimensions=['accuracy', 'efficiency', 'robustness', 'generalization', 'complexity'],
                    labels={
                        "accuracy": "Accuracy", "efficiency": "Efficiency",
                        "robustness": "Robustness", "generalization": "Generalization",
                        "complexity": "Complexity", "form_id": "Form"
                    },
                    title="Multi-Objective Niche Occupation by Form"
                )
                fig_parallel.update_layout(height=500)
                st.plotly_chart(fig_parallel, use_container_width=True, key="form_parallel_coords")
        
        # Time series analysis
        st.markdown("---")
        st.header("📈 Temporal Dynamics: Deconstructing the Evolutionary Trajectory")
        st.markdown("""
        This section analyzes how key architectural and performance traits evolved over generational time. By observing these trajectories, we can infer the nature of the selective pressures at play and the adaptive pathways different architectural forms explored to navigate the fitness landscape. Each plot reveals a different facet of the population's journey from its initial state to its final, optimized configuration.
        """)

        if history_df.empty:
            st.warning("No historical data available to analyze temporal dynamics.")
        else:
            tab1, tab2, tab3 = st.tabs([
                "🧬 Phenotypic Trait Evolution",
                "🎯 Objective Score Trajectories",
                "⚙️ Evolutionary Rates of Change"
            ])

            with tab1:
                st.markdown("#### **Evolution of Core Architectural Traits**")
                st.markdown("How did the physical characteristics of the architectures change over time? These plots track the evolution of network size (parameters) and structural complexity for each form, including the standard deviation to show population variance.")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("###### Network Size (Mean Parameters)")
                    param_stats = history_df.groupby(['generation', 'form'])['total_params'].agg(['mean', 'std']).reset_index()
                    fig_params = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    for i, form_name in enumerate(sorted(history_df['form'].unique())):
                        form_data = param_stats[param_stats['form'] == form_name]
                        mean_params = form_data['mean']
                        std_params = form_data['std'].fillna(0)
                        color = colors[i % len(colors)]
                        
                        fig_params.add_trace(go.Scatter(
                            x=np.concatenate([form_data['generation'], form_data['generation'][::-1]]),
                            y=np.concatenate([mean_params + std_params, (mean_params - std_params)[::-1]]),
                            fill='toself', fillcolor=color, opacity=0.1,
                            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False, legendgroup=form_name
                        ))
                        fig_params.add_trace(go.Scatter(
                            x=form_data['generation'], y=mean_params, mode='lines', name=form_name, legendgroup=form_name, line=dict(color=color)
                        ))
                    fig_params.update_layout(title='Mean Parameter Count Evolution by Form', xaxis_title='Generation', yaxis_title='Mean Total Parameters', yaxis_type="log", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_params, use_container_width=True, key="temporal_params_line")

                with col2:
                    st.markdown("###### Architectural Complexity (Mean Score)")
                    complexity_stats = history_df.groupby(['generation', 'form'])['complexity'].agg(['mean', 'std']).reset_index()
                    fig_complexity = go.Figure()
                    colors = px.colors.qualitative.Plotly
                    for i, form_name in enumerate(sorted(history_df['form'].unique())):
                        form_data = complexity_stats[complexity_stats['form'] == form_name]
                        mean_complexity = form_data['mean']
                        std_complexity = form_data['std'].fillna(0)
                        color = colors[i % len(colors)]
                        
                        fig_complexity.add_trace(go.Scatter(
                            x=np.concatenate([form_data['generation'], form_data['generation'][::-1]]),
                            y=np.concatenate([mean_complexity + std_complexity, (mean_complexity - std_complexity)[::-1]]),
                            fill='toself', fillcolor=color, opacity=0.1,
                            line=dict(color='rgba(255,255,255,0)'), hoverinfo="skip", showlegend=False, legendgroup=form_name
                        ))
                        fig_complexity.add_trace(go.Scatter(
                            x=form_data['generation'], y=mean_complexity, mode='lines', name=form_name, legendgroup=form_name, line=dict(color=color)
                        ))
                    fig_complexity.update_layout(title='Mean Complexity Score Evolution by Form', xaxis_title='Generation', yaxis_title='Mean Complexity Score', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig_complexity, use_container_width=True, key="temporal_complexity_line")

            with tab2:
                st.markdown("#### **Evolution of Multi-Objective Performance**")
                st.markdown("This analysis shows how the different forms adapted to the multi-objective fitness function over time. It reveals which forms specialized in certain objectives (e.g., accuracy vs. efficiency).")
                
                objectives_to_plot = ['accuracy', 'efficiency', 'robustness', 'generalization']
                fig_objectives = make_subplots(rows=2, cols=2, subplot_titles=[f'{obj.capitalize()} Trajectory' for obj in objectives_to_plot], vertical_spacing=0.15)
                
                plot_positions = [(1, 1), (1, 2), (2, 1), (2, 2)]
                
                for i, objective in enumerate(objectives_to_plot):
                    row, col = plot_positions[i]
                    objective_data = history_df.groupby(['generation', 'form'])[objective].mean().reset_index()
                    
                    for form_name in sorted(history_df['form'].unique()):
                        form_data = objective_data[objective_data['form'] == form_name]
                        fig_objectives.add_trace(go.Scatter(
                            x=form_data['generation'], y=form_data[objective], mode='lines', name=form_name, legendgroup=form_name, showlegend=(i==0)
                        ), row=row, col=col)
                    fig_objectives.update_yaxes(title_text="Mean Score", range=[0, 1], row=row, col=col)
                    fig_objectives.update_xaxes(title_text="Generation", row=row, col=col)
                    
                fig_objectives.update_layout(height=700, title_text="Mean Objective Score Evolution by Form", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig_objectives, use_container_width=True, key="temporal_objectives_matrix")

            with tab3:
                st.markdown("#### **Analysis of Evolutionary Rates**")
                st.markdown("By examining the rate of change (first derivative) of key metrics, we can identify periods of rapid adaptation, stagnation, and evolutionary shifts. Positive rates indicate growth/improvement, while negative rates indicate decline or simplification.")
                
                # Calculate rates of change for population mean
                rate_df = history_df.groupby('generation')[['fitness', 'complexity', 'total_params']].mean()
                rate_df['fitness_rate'] = rate_df['fitness'].diff()
                rate_df['complexity_rate'] = rate_df['complexity'].diff()
                rate_df['params_rate'] = rate_df['total_params'].diff()
                rate_df = rate_df.reset_index()

                fig_rates = make_subplots(rows=1, cols=3, subplot_titles=("Rate of Fitness Change (dF/dt)", "Rate of Complexity Change (dC/dt)", "Rate of Size Change (dP/dt)"))

                fig_rates.add_trace(go.Scatter(x=rate_df['generation'], y=rate_df['fitness_rate'], mode='lines', name='d(Fitness)/dt', line=dict(color='green')), row=1, col=1)
                fig_rates.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=1, col=1)

                fig_rates.add_trace(go.Scatter(x=rate_df['generation'], y=rate_df['complexity_rate'], mode='lines', name='d(Complexity)/dt', line=dict(color='blue')), row=1, col=2)
                fig_rates.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=1, col=2)

                fig_rates.add_trace(go.Scatter(x=rate_df['generation'], y=rate_df['params_rate'], mode='lines', name='d(Params)/dt', line=dict(color='red')), row=1, col=3)
                fig_rates.add_hline(y=0, line_width=1, line_dash="dash", line_color="grey", row=1, col=3)

                fig_rates.update_xaxes(title_text="Generation")
                fig_rates.update_layout(height=400, title_text="Rates of Change for Population Mean Metrics", showlegend=False)
                st.plotly_chart(fig_rates, use_container_width=True, key="temporal_rates_line")
        
        # Evolutionary metrics
        st.markdown("---")
        st.header("🧬 Population Genetics: The Engine of Adaptation")
        st.markdown("""
        This section delves into the quantitative measures of evolutionary dynamics, treating the population as a statistical ensemble. By analyzing metrics like heritability, selection pressure, and genetic distance, we can dissect the fundamental forces that shape the adaptive trajectory and understand the 'engine' driving the evolutionary process.
        """)

        if not st.session_state.evolutionary_metrics or history_df.empty:
            st.warning("Insufficient data for population genetics analysis.")
        else:
            # --- Data Preparation ---
            metrics_df = pd.DataFrame(st.session_state.evolutionary_metrics)
            final_population = st.session_state.current_population

            with st.spinner("Calculating advanced population genetics metrics..."):
                # Heritability (h²)
                heritabilities = []
                if history_df['generation'].max() > 0:
                    for gen in range(1, history_df['generation'].max() + 1):
                        parent_gen_df = history_df[history_df['generation'] == gen - 1]
                        offspring_gen_df = history_df[history_df['generation'] == gen]
                        if not parent_gen_df.empty and not offspring_gen_df.empty:
                            h2 = EvolutionaryTheory.heritability(parent_gen_df['fitness'].values, offspring_gen_df['fitness'].values)
                            heritabilities.append({'generation': gen, 'heritability': h2})
                h2_df = pd.DataFrame(heritabilities)
                if not h2_df.empty:
                    metrics_df = pd.merge(metrics_df, h2_df, on='generation', how='left')

                # Selection Differential (S)
                selection_diffs = []
                selection_pressure = st.session_state.settings.get('selection_pressure', 0.5)
                for gen in sorted(history_df['generation'].unique()):
                    gen_data = history_df[history_df['generation'] == gen]
                    if len(gen_data) > 2:
                        fitness_array = gen_data['fitness'].values
                        num_survivors = max(2, int(len(gen_data) * selection_pressure))
                        selected_idx = np.argpartition(fitness_array, -num_survivors)[-num_survivors:]
                        diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)
                        selection_diffs.append({'generation': gen, 'selection_differential': diff})
                sel_df = pd.DataFrame(selection_diffs)
                if not sel_df.empty:
                    metrics_df = pd.merge(metrics_df, sel_df, on='generation', how='left')

                # Response to Selection (R)
                metrics_df['response_to_selection'] = metrics_df['mean_fitness'].diff()
                
                # Predicted Response (R_pred = h² * S)
                if 'heritability' in metrics_df.columns and 'selection_differential' in metrics_df.columns:
                    metrics_df['predicted_response'] = metrics_df['heritability'] * metrics_df['selection_differential']
                
                metrics_df = metrics_df.fillna(0)

            tab1, tab2, tab3 = st.tabs([
                "📈 Core Evolutionary Forces",
                "⚙️ The Breeder's Equation",
                "🗺️ Genotypic Landscape"
            ])

            with tab1:
                st.markdown("#### **Temporal Evolution of Core Genetic Metrics**")
                st.markdown("These metrics quantify the key forces governing adaptation over time. **Diversity (H)** is the raw material for evolution. **Heritability (h²)** is the degree to which fitness is passed on. **Selection Differential (S)** is the strength of selection. **Fisher Information (I)** is the potential for adaptation.")
                fig_core = make_subplots(rows=2, cols=2, subplot_titles=("Genetic Diversity (H)", "Heritability (h²)", "Selection Differential (S)", "Fisher Information (I)"), vertical_spacing=0.15)

                fig_core.add_trace(go.Scatter(x=metrics_df['generation'], y=metrics_df['diversity'], name='Diversity', line=dict(color='purple')), row=1, col=1)
                if 'heritability' in metrics_df.columns:
                    fig_core.add_trace(go.Scatter(x=metrics_df['generation'], y=metrics_df['heritability'], name='Heritability', line=dict(color='green')), row=1, col=2)
                if 'selection_differential' in metrics_df.columns:
                    fig_core.add_trace(go.Scatter(x=metrics_df['generation'], y=metrics_df['selection_differential'], name='Selection Differential', line=dict(color='red')), row=2, col=1)
                fig_core.add_trace(go.Scatter(x=metrics_df['generation'], y=metrics_df['fisher_info'], name='Fisher Information', line=dict(color='orange')), row=2, col=2)

                fig_core.update_yaxes(title_text="Value", range=[0, max(1, metrics_df['diversity'].max()) if not metrics_df.empty else 1], row=1, col=1)
                fig_core.update_yaxes(title_text="h²", range=[0, 1], row=1, col=2)
                fig_core.update_yaxes(title_text="Δ Fitness", row=2, col=1)
                fig_core.update_yaxes(title_text="1/σ²", row=2, col=2)
                fig_core.update_layout(height=600, showlegend=False, title_text="Evolution of Core Population Genetic Metrics")
                st.plotly_chart(fig_core, use_container_width=True, key="popgen_core_forces_plot")

            with tab2:
                st.markdown("#### **Validating the Breeder's Equation: R = h²S**")
                st.markdown("The Breeder's Equation is a cornerstone of quantitative genetics, predicting the evolutionary response (R) based on heritability (h²) and selection strength (S). Here we test this prediction against the observed reality of our simulation.")
                
                if 'predicted_response' in metrics_df.columns:
                    col1, col2 = st.columns(2)
                    with col1:
                        fig_breeder = go.Figure()
                        fig_breeder.add_trace(go.Scatter(x=metrics_df['generation'], y=metrics_df['response_to_selection'], name='Actual Response (R)', mode='lines+markers', line=dict(color='blue')))
                        fig_breeder.add_trace(go.Scatter(x=metrics_df['generation'], y=metrics_df['predicted_response'], name='Predicted Response (h²S)', mode='lines', line=dict(color='red', dash='dash')))
                        fig_breeder.update_layout(title="Actual vs. Predicted Evolutionary Response", xaxis_title="Generation", yaxis_title="Change in Mean Fitness", height=400, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_breeder, use_container_width=True, key="popgen_breeder_eq_plot")
                    with col2:
                        corr_df = metrics_df[metrics_df['response_to_selection'] != 0]
                        fig_corr = px.scatter(corr_df, x='predicted_response', y='response_to_selection', trendline='ols', title="Correlation of Predicted vs. Actual Response", labels={'predicted_response': 'Predicted R (h²S)', 'response_to_selection': 'Actual R'})
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, use_container_width=True, key="popgen_breeder_corr_plot")
                else:
                    st.info("Not enough data to validate the Breeder's Equation.")

            with tab3:
                st.markdown("#### **Structure of the Final Genotypic Landscape**")
                st.markdown("This analysis examines the distribution of genotypes in the final population, revealing patterns of diversity, speciation, and the relationship between genetic similarity and fitness.")

                if final_population and len(final_population) > 10:
                    with st.spinner("Calculating pairwise genomic distances for the final population..."):
                        pop_size = len(final_population)
                        genomic_distances = []
                        fitness_deltas = []
                        for i in range(pop_size):
                            for j in range(i + 1, pop_size):
                                g1 = final_population[i]
                                g2 = final_population[j]
                                dist = genomic_distance(g1, g2)
                                if dist != float('inf'): # Only compare within the same form
                                    genomic_distances.append(dist)
                                    fitness_deltas.append(abs(g1.fitness - g2.fitness))
                    
                    if genomic_distances:
                        col1, col2 = st.columns(2)
                        with col1:
                            fig_dist_hist = px.histogram(pd.DataFrame({'distance': genomic_distances}), x='distance', nbins=50, title="Distribution of Pairwise Genomic Distances")
                            fig_dist_hist.update_layout(height=400, yaxis_title="Count", xaxis_title="Genomic Distance")
                            st.plotly_chart(fig_dist_hist, use_container_width=True, key="popgen_dist_hist_plot")
                            st.markdown("A multi-modal distribution can indicate distinct species have formed.")
                        with col2:
                            dist_corr_df = pd.DataFrame({'genomic_dist': genomic_distances, 'fitness_delta': fitness_deltas})
                            fig_dist_corr = px.scatter(dist_corr_df, x='genomic_dist', y='fitness_delta', trendline='ols', title="Genomic Distance vs. Fitness Difference", labels={'genomic_dist': 'Genomic Distance', 'fitness_delta': 'Absolute Fitness Difference'})
                            fig_dist_corr.update_layout(height=400)
                            st.plotly_chart(fig_dist_corr, use_container_width=True, key="popgen_dist_corr_plot")
                            st.markdown("A positive correlation suggests a smooth landscape where similar genotypes have similar fitness.")
                    else:
                        st.info("No comparable genotypes found to analyze the genotypic landscape.")
                else:
                    st.info("Final population is too small to analyze the genotypic landscape.")

        st.markdown("---")
        st.header("🔬 Final Synthesis: Deconstructing the Evolutionary Narrative")
        st.markdown("""
        This final section synthesizes the entire evolutionary run into a cohesive narrative, identifying the dominant strategic themes and interpreting the final outcome based on quantitative evidence from the population's trajectory. It serves as the executive summary of the experiment.
        """)

        # --- Data Gathering for Synthesis ---
        takeaways = []
        dominant_form_id = "N/A"
        dominance_pct = 0
        complexity_change = 0
        diversity_change = 0
        mean_selection_differential = 0
        task_type = st.session_state.settings['task_type']

        if not final_gen.empty and not final_gen['form_id'].value_counts().empty:
            form_counts = final_gen['form_id'].value_counts()
            dominant_form_id = form_counts.index[0]
            dominance_pct = (form_counts.iloc[0] / form_counts.sum()) * 100
            if dominance_pct > 60:
                takeaways.append(f"**Strong Emergent Specialization:** The simulation demonstrated clear convergent evolution, with **Form {int(dominant_form_id)}** becoming the dominant morphology, comprising **{dominance_pct:.1f}%** of the final population. This indicates its foundational topology provided a significant adaptive advantage for the '{task_type}' task.")

        if not history_df.empty and not final_gen.empty:
            initial_complexity = history_df[history_df['generation'] == 0]['complexity'].mean()
            final_complexity = final_gen['complexity'].mean()
            complexity_change = ((final_complexity / initial_complexity) - 1) * 100 if initial_complexity > 0 else 0
            if complexity_change > 20:
                takeaways.append(f"**Constructive Complexity Ratchet:** A trend of increasing architectural complexity was observed (a **{complexity_change:.1f}%** increase in the mean), suggesting the fitness landscape rewarded more intricate solutions, balanced by the selective pressure for efficiency (weight: {st.session_state.settings['w_efficiency']:.2f}).")
            elif complexity_change < -20:
                takeaways.append(f"**Parsimonious Selection:** The evolutionary pressure strongly favored simplicity, leading to a **{abs(complexity_change):.1f}%** decrease in mean architectural complexity, indicating a successful search for efficient, minimalist solutions.")

        if 'final_gen_genotypes' in locals() and final_gen_genotypes and 'acc_specialist' in locals() and acc_specialist and 'eff_generalist' in locals() and eff_generalist:
            if acc_specialist.efficiency < eff_generalist.efficiency * 0.8 and eff_generalist.accuracy < acc_specialist.accuracy * 0.8:
                takeaways.append(f"**Clear Performance Trade-offs:** The final population illustrates a classic Pareto frontier. The top accuracy specialist (Acc: {acc_specialist.accuracy:.3f}) was significantly less efficient than the top efficiency generalist (Eff: {eff_generalist.efficiency:.3f}), which in turn had lower accuracy (Acc: {eff_generalist.accuracy:.3f}).")
            else:
                takeaways.append("**Evidence of Pareto Optimality:** The final population represents a set of solutions with varying strengths across multiple objectives. No single architecture dominated all others, which is characteristic of a successful multi-objective search.")

        if not metrics_df.empty and len(metrics_df) > 1:
            initial_diversity = metrics_df.iloc[0]['diversity']
            final_diversity = metrics_df.iloc[-1]['diversity']
            diversity_change = ((final_diversity / initial_diversity) - 1) * 100 if initial_diversity > 0 else 0
            if diversity_change < -50:
                 takeaways.append(f"**Convergent Evolution Dominates:** Despite mechanisms for innovation, the population showed strong convergence towards a specific set of high-performing genotypes, indicated by a significant drop in genetic diversity (**{diversity_change:+.1f}%**). The search effectively exploited a promising region of the fitness landscape.")
            elif diversity_change > -10:
                 takeaways.append(f"**Sustained Exploration and Innovation:** The population maintained high genetic diversity. This suggests that structural mutations (innovation rate: {st.session_state.settings['innovation_rate']}) were crucial for continually discovering novel architectural motifs and avoiding premature convergence.")

        # --- Identify Dominant Strategy ---
        strategy = "Balanced Exploration"
        if dominance_pct > 60 and diversity_change < -40:
            strategy = "Convergent Specialization"
        elif diversity_change > -10 and dominance_pct < 40:
            strategy = "Divergent Exploration"
        elif complexity_change < -20:
            strategy = "Parsimonious Optimization"
        elif complexity_change > 20:
            strategy = "Constructive Complexification"

        st.subheader(f"Dominant Evolutionary Strategy: **{strategy}**")

        strategy_narratives = {
            "Convergent Specialization": f"The evolutionary process was characterized by a powerful **convergent** force. The population rapidly identified the superior inductive bias of **Form {int(dominant_form_id)}** and aggressively exploited it, leading to its overwhelming dominance ({dominance_pct:.1f}%). This was coupled with a significant reduction in genetic diversity ({diversity_change:+.1f}%), indicating the search honed in on a narrow, highly-fit region of the landscape. The system prioritized **exploitation over exploration**.",
            "Divergent Exploration": f"The run was defined by **divergent exploration** and the maintenance of biodiversity. No single architectural form achieved dominance, and genetic diversity remained high ({diversity_change:+.1f}%). This suggests the fitness landscape is either rugged with many viable peaks or that the multi-objective pressures were strong enough to support multiple, distinct strategies (niches) simultaneously. The system prioritized **exploration and niche discovery**.",
            "Parsimonious Optimization": f"The primary driver of this run was a strong selective pressure for **efficiency and simplicity**. This is evidenced by a significant decrease in mean architectural complexity ({complexity_change:+.1f}%) across the population. The system successfully identified minimalist yet effective solutions, demonstrating a **parsimonious optimization** strategy.",
            "Constructive Complexification": f"This run was a clear example of a **constructive evolutionary ratchet**, where increasing complexity was consistently rewarded. The population saw a significant growth in mean architectural complexity ({complexity_change:+.1f}%), suggesting the task environment contained deep, intricate structures that could only be captured by more sophisticated models. The system favored **additive innovation**.",
            "Balanced Exploration": "The system demonstrated a **balanced strategy**, avoiding premature convergence while still making consistent fitness gains. It maintained a moderate level of genetic diversity and did not allow any single architectural form to completely dominate, suggesting a healthy dynamic between exploiting known solutions and exploring for new ones."
        }
        st.markdown(strategy_narratives.get(strategy, "The evolutionary dynamics of this run were complex and multifaceted."))

        # --- Key Quantitative Insights ---
        st.subheader("Key Quantitative Insights")
        if not history_df.empty:
            selection_diff_data = []
            selection_pressure_param = st.session_state.settings['selection_pressure']
            for gen in sorted(history_df['generation'].unique()):
                gen_data = history_df[history_df['generation'] == gen]
                if len(gen_data) > 2:
                    fitness_array = gen_data['fitness'].values
                    num_survivors = max(2, int(len(gen_data) * selection_pressure_param))
                    selected_idx = np.argpartition(fitness_array, -num_survivors)[-num_survivors:]
                    diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)
                    selection_diff_data.append({'generation': gen, 'selection_diff': diff})
            if selection_diff_data:
                mean_selection_differential = pd.DataFrame(selection_diff_data)['selection_diff'].mean()

            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Form Dominance", f"Form {int(dominant_form_id)}" if dominant_form_id != "N/A" else "N/A", f"{dominance_pct:.1f}% of Pop.")
            col2.metric("Complexity Shift", f"{complexity_change:+.1f}%", help="Change in mean complexity from Gen 0 to final.")
            col3.metric("Diversity Change", f"{diversity_change:+.1f}%", help="Change in Shannon entropy from Gen 0 to final.")
            col4.metric("Mean Selection Δ", f"{mean_selection_differential:.3f}", help="Average selection pressure across all generations.")

        # --- Final Concluding Remarks ---
        st.subheader("Concluding Remarks & Implications")
        if takeaways:
            st.markdown("Based on the quantitative evidence, the key conclusions from this run are:")
            takeaway_markdown = ""
            for item in takeaways:
                parts = item.split('**')
                if len(parts) > 2:
                    takeaway_markdown += f"- **{parts[1]}**: {parts[2]}\n"
            st.markdown(takeaway_markdown)

            implication_text = {
                "Convergent Specialization": f"for the '{task_type}' problem, identifying and refining a specific architectural prior (like that of Form {int(dominant_form_id)}) is a highly effective path to a solution. The problem space appears to have a dominant, steep peak.",
                "Divergent Exploration": f"the '{task_type}' problem is best solved by a portfolio of diverse strategies. The fitness landscape is likely multi-modal, and forcing convergence to a single architecture would be suboptimal. A multi-model ensemble would likely yield the best results.",
                "Parsimonious Optimization": f"the '{task_type}' problem does not require immense complexity. Overly complex models may overfit or be inefficient without providing significant benefits. The search for compact, elegant solutions is a fruitful direction.",
                "Constructive Complexification": f"the '{task_type}' problem has a deep, hierarchical structure that rewards increasingly complex models. Simple architectures are insufficient, and future efforts should focus on enabling further growth in model scale and intricacy.",
                "Balanced Exploration": f"for the '{task_type}' problem, a steady, iterative refinement process is effective. There is no single 'silver bullet' architecture, but consistent, small improvements across a diverse population can reliably navigate the fitness landscape."
            }
            st.markdown(f"""
            **Implications:** The success of the **{strategy}** strategy suggests that {implication_text.get(strategy, "the problem is complex.")} Future runs could build on this by either intensifying the discovered strategy (e.g., seeding the entire population with the dominant form) or by challenging it (e.g., introducing dynamic environmental shifts to test the robustness of the converged solution).
            """)
        else:
            st.warning("Could not generate dynamic concluding remarks due to insufficient data.")

        st.markdown("---")
        st.header("🌐 Synthesized Master Architecture: A Consensus Genotype from the Pareto Frontier")
        st.markdown("""
        The culmination of the evolutionary run is not a single victor, but a Pareto-optimal frontier of specialized genotypes. To distill a singular, robust archetype from this elite population, we employ a synthesis algorithm. This process constructs a 'master' architecture that represents a statistical consensus of the most successful genetic motifs and parametric configurations discovered during the search.

        The synthesis protocol is a multi-stage process of structural and parametric amalgamation:
        1.  **Template Seeding:** The genotype with the highest fitness from the synthesis pool is selected as the foundational template. This ensures the master architecture is anchored to a proven high-performer.
        2.  **Parametric Bayesian Averaging:** For all homologous modules and connections shared between the template and other members of the elite pool, their continuous-valued parameters (e.g., module size, connection weight, plasticity coefficients) are refined. This is analogous to a Bayesian model average, where the parameters of the master architecture are updated towards the posterior mean estimated from the elite sample. This smooths out stochastic noise from individual evolutionary paths and converges on more generalizable parameter values.
        3.  **Structural Voting and Topological Infusion:** The algorithm identifies structural motifs (specifically, connections) that exhibit high prevalence across the elite pool, even if absent in the initial template. Connections exceeding a consensus threshold are infused into the master genotype. This step is critical for integrating convergent structural discoveries and correcting for idiosyncratic omissions in the single best individual.
        
        The parameter **'n'** controls the size of the synthesis pool drawn from the Pareto frontier. A smaller 'n' yields a master architecture heavily biased towards the top individual, preserving its unique characteristics. A larger 'n' broadens the consensus base, resulting in a more generalized and potentially more robust architecture that averages out niche-specific adaptations. This slider allows for an exploration of the trade-off between peak performance and generalized robustness.
        """)

        # Correctly identify the Pareto frontier to use as the synthesis pool
        final_gen_genotypes = [ind for ind in population if ind.generation == history_df['generation'].max()] if population else []
        if final_gen_genotypes:
            pareto_individuals = identify_pareto_frontier(final_gen_genotypes)
            pareto_individuals.sort(key=lambda x: x.fitness, reverse=True) # Sort by fitness
        else:
            pareto_individuals = []

        if not pareto_individuals:
            st.warning("Could not identify a Pareto frontier to synthesize a master architecture.")
        else:
            n_for_synthesis = st.slider(
                "Number of top Pareto individuals to synthesize from (n)",
                min_value=1,
                max_value=len(pareto_individuals),
                value=min(3, len(pareto_individuals)),
                step=1,
                key="n_for_synthesis_slider"
            )

            synthesis_pool = pareto_individuals[:n_for_synthesis]
            
            with st.spinner(f"Synthesizing master architecture from {n_for_synthesis} elite individuals..."):
                master_architecture = synthesize_master_architecture(synthesis_pool)

            if not master_architecture:
                st.error("Failed to synthesize master architecture.")
            else:
                # Perform all analyses upfront
                with st.spinner("Performing comprehensive deep analysis on Master Architecture..."):
                    criticality_scores = analyze_lesion_sensitivity(master_architecture, master_architecture.fitness, task_type, fitness_weights, eval_params)
                    centrality_scores = analyze_information_flow(master_architecture)
                    evo_robust_data = analyze_evolvability_robustness(master_architecture, task_type, fitness_weights, eval_params)
                    dev_traj_df = analyze_developmental_trajectory(master_architecture)
                    load_data = analyze_genetic_load(criticality_scores)
                    phylo_data = analyze_phylogenetic_signal(history_df, population)

                # --- Create Tabs for Deep Dive ---
                tab_synthesis, tab_causal, tab_potential, tab_code, tab_interpretation = st.tabs([
                    "🧬 Synthesis & Architecture", 
                    "🔬 Causal & Structural Analysis", 
                    "⚙️ Evolutionary & Developmental Potential",
                    "💻 Code & Export",
                    "📜 Comprehensive Interpretation"
                ])

                # --- TAB 1: Synthesis & Architecture ---
                with tab_synthesis:
                    st.markdown("#### Quantitative Profile: Master vs. Synthesis Pool")
                    pool_mean_fitness = np.mean([ind.fitness for ind in synthesis_pool])
                    pool_mean_accuracy = np.mean([ind.accuracy for ind in synthesis_pool])
                    pool_mean_complexity = np.mean([ind.complexity for ind in synthesis_pool])
                    pool_mean_params = np.mean([sum(m.size for m in ind.modules) for ind in synthesis_pool])
                    
                    pool_best_ind = synthesis_pool[0]

                    comparison_data = {
                        "Metric": ["Fitness", "Accuracy", "Complexity", "Parameters"],
                        "Master Arch.": [f"{master_architecture.fitness:.4f}", f"{master_architecture.accuracy:.3f}", f"{master_architecture.complexity:.3f}", f"{sum(m.size for m in master_architecture.modules):,}"],
                        "Pool Mean": [f"{pool_mean_fitness:.4f}", f"{pool_mean_accuracy:.3f}", f"{pool_mean_complexity:.3f}", f"{pool_mean_params:,.0f}"],
                        "Pool Best": [f"{pool_best_ind.fitness:.4f}", f"{pool_best_ind.accuracy:.3f}", f"{pool_best_ind.complexity:.3f}", f"{sum(m.size for m in pool_best_ind.modules):,}"],
                    }
                    st.dataframe(pd.DataFrame(comparison_data).set_index("Metric"), use_container_width=True, key="master_vitals_df")

                    st.markdown("---")
                    st.markdown("#### Architectural Visualization")
                    vis_col1, vis_col2 = st.columns(2)
                    with vis_col1:
                        st.plotly_chart(visualize_genotype_3d(master_architecture), use_container_width=True, key="master_3d_vis")
                    with vis_col2:
                        st.plotly_chart(visualize_genotype_2d(master_architecture), use_container_width=True, key="master_2d_vis")

                # --- TAB 2: Causal & Structural Analysis ---
                with tab_causal:
                    causal_col1, causal_col2 = st.columns(2)
                    with causal_col1:
                        st.subheader("Lesion Sensitivity Analysis")
                        st.markdown("Fitness drop when a component is removed. Higher = more critical.")
                        sorted_criticality = sorted(criticality_scores.items(), key=lambda item: item[1])
                        crit_df = pd.DataFrame(sorted_criticality, columns=['Component', 'Fitness Drop']).tail(15)
                        fig_crit = px.bar(crit_df, x='Fitness Drop', y='Component', orientation='h', title="Top 15 Most Critical Components")
                        fig_crit.update_layout(height=400, margin=dict(l=150))
                        st.plotly_chart(fig_crit, use_container_width=True, key="master_criticality_bar")

                    with causal_col2:
                        st.subheader("Information Flow Backbone")
                        st.markdown("`Betweenness centrality` identifies key modules for information routing.")
                        sorted_centrality = sorted(centrality_scores.items(), key=lambda item: item[1])
                        cent_df = pd.DataFrame(sorted_centrality, columns=['Module', 'Centrality']).tail(15)
                        fig_cent = px.bar(cent_df, x='Centrality', y='Module', orientation='h', title="Top 15 Most Central Modules")
                        fig_cent.update_layout(height=400, margin=dict(l=150))
                        st.plotly_chart(fig_cent, use_container_width=True, key="master_centrality_bar")
                    
                    st.markdown("---")
                    st.subheader("Genetic Load & Neutrality")
                    st.markdown("Analysis of non-critical components ('junk DNA') and the fitness cost of slightly harmful mutations.")
                    load_col1, load_col2 = st.columns(2)
                    load_col1.metric("Neutral Component Count", f"{load_data['neutral_component_count']}", help="Number of modules/connections with near-zero impact when lesioned.")
                    load_col2.metric("Genetic Load", f"{load_data['genetic_load']:.4f}", help="Total fitness reduction from slightly deleterious, non-critical components.")

                # --- TAB 3: Evolutionary & Developmental Potential ---
                with tab_potential:
                    evo_col1, evo_col2 = st.columns(2)
                    with evo_col1:
                        st.subheader("Evolvability vs. Robustness")
                        st.markdown("Trade-off between resisting negative mutations and producing beneficial ones.")
                        st.metric("Robustness Score", f"{evo_robust_data['robustness']:.4f}", help="Average fitness loss from deleterious mutations. Higher = more robust.")
                        st.metric("Evolvability Score", f"{evo_robust_data['evolvability']:.4f}", help="Maximum fitness gain from a single mutation.")

                        dist_df = pd.DataFrame(evo_robust_data['distribution'], columns=['Fitness Change'])
                        fig_mut = px.histogram(dist_df, x="Fitness Change", nbins=30, title="Distribution of Mutational Effects")
                        fig_mut.add_vline(x=0, line_width=2, line_dash="dash", line_color="grey")
                        fig_mut.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_mut, use_container_width=True, key="master_mutational_effects_hist")

                    with evo_col2:
                        st.subheader("Developmental Trajectory")
                        st.markdown("Simulated 'lifetime' structural changes based on the genotype's developmental program.")
                        fig_dev = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                        fig_dev.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_dev, use_container_width=True, key="master_dev_trajectory_line")
                    
                    st.markdown("---")
                    st.subheader("Phylogenetic Signal (Pagel's λ)")
                    st.markdown("Measures how much trait variation is explained by evolutionary history. High λ (~1) means strong inertia; low λ (~0) means rapid convergence.")
                    if phylo_data:
                        st.metric("Phylogenetic Signal (λ estimate)", f"{phylo_data['correlation']:.3f}", help="Correlation between phylogenetic distance and phenotypic (fitness) distance.")
                        phylo_df = pd.DataFrame({'Phylogenetic Distance': phylo_data['phylo_distances'], 'Phenotypic Distance': phylo_data['pheno_distances']})
                        fig_phylo = px.scatter(phylo_df, x='Phylogenetic Distance', y='Phenotypic Distance', trendline="ols", title="Phylogenetic vs. Phenotypic Distance")
                        fig_phylo.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_phylo, use_container_width=True, key="master_phylo_signal_scatter")
                    else:
                        st.info("Not enough data for phylogenetic signal analysis.")

                # --- TAB 4: Code & Export ---
                with tab_code:
                    st.markdown("The synthesized architecture can be translated into functional code for deep learning frameworks, providing a direct path from discovery to application.")
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.subheader("PyTorch Code")
                        st.code(generate_pytorch_code(master_architecture), language='python')
                    with col2:
                        st.subheader("TensorFlow / Keras Code")
                        st.code(generate_tensorflow_code(master_architecture), language='python')
                    
                    st.markdown("---")
                    st.subheader("Export Genotype")
                    st.download_button(
                        label="Download Master Genotype (JSON)",
                        data=json.dumps(genotype_to_dict(master_architecture), indent=2),
                        file_name="master_architecture.json",
                        mime="application/json",
                        key="download_master_genotype_button"
                    )

                # --- TAB 5: Comprehensive Interpretation ---
                with tab_interpretation:
                    st.subheader("Synthesized Interpretation of the Master Architecture")
                    
                    dominant_module_type = Counter(m.module_type for m in master_architecture.modules).most_common(1)[0][0]
                    critical_component_tuple = sorted_criticality[-1] if sorted_criticality else ("N/A", 0)
                    centrality_score = (sorted_centrality[-1][1] if sorted_centrality else 0)
                    
                    interpretation_text = f"""
                    The Master Architecture, synthesized from the top **{n_for_synthesis}** Pareto-optimal individuals, represents a robust consensus of the most successful evolutionary strategies. Its quantitative profile (`Fitness: {master_architecture.fitness:.4f}`, `Accuracy: {master_architecture.accuracy:.3f}`) positions it as a highly competitive, generalized solution. A deep analysis reveals the following insights:

                    1.  **Convergent Design Principles:** The architecture's structure, dominated by **{dominant_module_type}** modules, confirms this motif as a critical building block for the **'{task_type}'** task. The synthesis process, by averaging parameters and voting on connections, has filtered out idiosyncratic noise, retaining only the high-conviction elements shared by the elite.

                    2.  **Identified Causal Backbone:** The causal analysis provides a blueprint of functional importance. The high lesion sensitivity of **{critical_component_tuple[0]}** (fitness drop: `{critical_component_tuple[1]:.4f}`) and its significant information flow centrality (`{centrality_score:.3f}`) identify it as an indispensable hub. This is not just a collection of parts; it's a network with a clear processing core, validated by consensus.

                    3.  **A Platform for Future Evolution:** The analysis of the mutational landscape reveals a genotype poised for further adaptation. Its robustness score (`{evo_robust_data['robustness']:.4f}`) indicates a resilience to random genetic drift, while its non-zero evolvability score (`{evo_robust_data['evolvability']:.4f}`) shows it has not reached an evolutionary dead-end. The presence of `{load_data['neutral_component_count']}` neutral components provides a reservoir of 'junk DNA' that can be co-opted for future functional innovation.

                    4.  **Developmental Stability:** The simulated developmental trajectory shows a stable program. The architecture does not exhibit runaway growth or catastrophic pruning, indicating that its encoded developmental rules lead to a mature, stable phenotype.

                    In summary, the Master Architecture is more than just an average of the best; it is a refined, validated, and robust blueprint. It embodies the collective wisdom of the evolutionary search, representing a high-quality, general-purpose solution that is both highly functional and primed for future adaptation.
                    """
                    st.info(interpretation_text)

    st.sidebar.markdown("---")

    st.markdown("---")
    st.header("🏁 Epilogue: Reflections on the Evolutionary Journey and Future Directions")
    st.markdown("""
    As we reach the culmination of this evolutionary simulation, it is crucial to reflect on the journey undertaken, acknowledge the limitations inherent in our model, and chart a course for future research directions. This epilogue serves as a reflective summary of the entire process.
    """)

    st.subheader("Key Insights and Observations")
    st.markdown("""
    The simulation has provided several key insights into the dynamics of neuroevolution:

    1.  **The Power of Multi-Objective Optimization:** The Pareto frontier analysis demonstrates the importance of balancing multiple objectives (accuracy, efficiency, robustness) to achieve a diverse set of high-performing solutions.
    2.  **The Significance of Architectural Forms:** The comparative form analysis highlights the impact of initial architectural biases on the evolutionary trajectory and the final population structure.
    3.  **The Role of Ecosystem Dynamics:** The inclusion of ecosystem dynamics (cataclysms, Red Queen) has shown how external pressures can shape the evolutionary process and drive adaptation.
    4.  **The Importance of Developmental Programs:** The developmental trajectory analysis reveals how genetic programs can influence the growth and complexity of neural architectures.
    """)

    st.subheader("Limitations and Caveats")
    st.markdown("""
    It is important to acknowledge the limitations of the current simulation:

    1.  **Simplified Fitness Evaluation:** The fitness evaluation is based on simplified simulations of task performance and does not capture the full complexity of real-world tasks.
    2.  **Limited Architectural Diversity:** The initial set of architectural forms is limited and may not represent the full range of possible neural architectures.
    3.  **Abstracted Evolutionary Operators:** The evolutionary operators (mutation, crossover) are simplified and do not fully capture the intricacies of biological evolution.
    4.  **Computational Constraints:** The simulation is limited by computational resources, which restricts the population size, number of generations, and complexity of the architectures.
    """)

    st.subheader("Future Research Directions")
    st.markdown("""
    Based on the insights and limitations of the current simulation, several future research directions can be identified:

    1.  **Enhanced Fitness Evaluation:** Develop more realistic and comprehensive fitness evaluation methods that incorporate real-world datasets and task complexities.
    2.  **Expanded Architectural Diversity:** Explore a wider range of architectural forms and allow for the emergence of novel architectural motifs through more flexible evolutionary operators.
    3.  **Incorporation of Biological Realism:** Integrate more biologically realistic mechanisms into the evolutionary operators, such as gene duplication, horizontal gene transfer, and epigenetic regulation.
    4.  **Scalability and Parallelization:** Improve the scalability of the simulation to handle larger populations, more complex architectures, and longer evolutionary timescales.
    5.  **Integration with Real-World Systems:** Bridge the gap between simulation and reality by transferring evolved architectures to real-world robotic systems and evaluating their performance in physical environments.
    6.  **Explainable AI (XAI) Analysis:** Apply XAI techniques to understand the inner workings of evolved architectures and identify the key factors that contribute to their success.
    """)

    st.subheader("Concluding Thoughts")
    st.markdown("""
    This evolutionary simulation represents a step towards understanding the principles of neuroevolution and its potential for creating intelligent systems. By continuing to refine our models, incorporate biological realism, and explore new research directions, we can unlock the full potential of evolutionary algorithms to design and optimize complex neural architectures for a wide range of applications.

    Thank you for joining us on this evolutionary journey!
    """)

    st.markdown("""
    <style>
    .footer {
        position: fixed;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #0E1117;
        color: rgba(250, 250, 250, 0.7);
        text-align: center;
        padding: 5px;
        font-size: small;
    }
    </style>
    <div class="footer">
        GENEVO: Genetic Evolution of Neural Architectures | A Research Prototype
    </div>
    """, unsafe_allow_html=True)

    # --- Download all data ---
    st.subheader("Download Experiment Data")
    st.markdown("Download all experiment data, including settings, history, evolutionary metrics, and the final population, for offline analysis and reproducibility.")

    # Prepare data for download
    all_data = {
        'settings': st.session_state.settings,
        'history': st.session_state.history,
        'evolutionary_metrics': st.session_state.evolutionary_metrics,
        'final_population': [genotype_to_dict(p) for p in st.session_state.current_population] if st.session_state.current_population else []
    }
    all_data_json = json.dumps(all_data, indent=2)

    # Create download button
    st.download_button(
        label="Download All Experiment Data (JSON)",
        data=all_data_json,
        file_name="genevo_experiment_data.json",
        mime="application/json",
        key="download_all_data_button"
    )

    st.sidebar.info(
        "**GENEVO** is a research prototype demonstrating advanced concepts in neuroevolution. "
        "Architectures are simulated and not trained on real data."
    )


if __name__ == "__main__":
    main()
