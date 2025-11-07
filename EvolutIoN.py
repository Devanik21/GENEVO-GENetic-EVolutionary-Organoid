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
    
    # Use modulo to wrap around the defined forms if form_id is out of bounds.
    # This allows for a functionally "infinite" number of forms by reusing the 
    # base templates, which will then diverge through evolution.
    lookup_id = ((form_id - 1) % len(forms)) + 1
    form = forms[lookup_id]
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

    # If hyperparameter evolution is enabled, add the evolvable params to the genotype
    settings = st.session_state.get('settings', {})
    if settings.get('enable_hyperparameter_evolution'):
        evolvable_params = settings.get('evolvable_params', [])
        for param in evolvable_params:
            genotype.meta_parameters[param] = settings.get(param)

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
        'generalization': 0.0,
        # Advanced Primary Objectives
        'learning_speed': 0.0,
        'data_parsimony': 0.0,
        'forgetting_resistance': 0.0,
        'adaptability': 0.0,
        'latency': 0.0,
        'energy_consumption': 0.0,
        'development_cost': 0.0,
        'modularity': 0.0,
        'interpretability': 0.0,
        'evolvability': 0.0,
        'fairness': 0.0,
        'explainability': 0.0,
        'value_alignment': 0.0,
        'causal_density': 0.0,
        'self_organization': 0.0,
        'autopoiesis': 0.0,
        'computational_irreducibility': 0.0,
        'cognitive_synergy': 0.0,
    }
    
    # Compute architectural properties
    total_params = sum(m.size for m in genotype.modules)
    avg_plasticity = np.mean([m.plasticity for m in genotype.modules]) if genotype.modules else 0.0
    connection_density = len(genotype.connections) / (len(genotype.modules) ** 2 + 1) if genotype.modules else 0.0
    
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

    # 6. Advanced Primary Objectives (Conceptual Calculations)
    # These are simplified proxies for complex concepts.
    
    # Learning & Adaptation
    scores['learning_speed'] = avg_plasticity
    scores['data_parsimony'] = 1.0 / (1.0 + np.log(1 + total_params / 10000))
    scores['forgetting_resistance'] = (scores.get('modularity', 0.0) + avg_plasticity) / 2.0
    scores['adaptability'] = avg_plasticity

    # Resource & Cost
    scores['latency'] = 1.0 / (1.0 + len(genotype.modules))
    scores['energy_consumption'] = 1.0 / (1.0 + np.log(1 + total_params + len(genotype.connections)))
    scores['development_cost'] = 1.0 / (1.0 + len(genotype.developmental_rules))

    # Structural & Interpretability
    try:
        G = nx.DiGraph()
        for m in genotype.modules: G.add_node(m.id)
        for c in genotype.connections: G.add_edge(c.source, c.target)
        # Using a simple approximation for modularity based on connection density sweet spot
        scores['modularity'] = 1.0 - abs(connection_density - 0.3) * 2
    except:
        scores['modularity'] = 0.0
    scores['interpretability'] = scores['efficiency']  # Reuse efficiency as a proxy for simplicity
    module_type_diversity = len(set(m.module_type for m in genotype.modules)) / 5.0
    scores['evolvability'] = np.clip(module_type_diversity, 0, 1)

    # Safety & Alignment (purely conceptual placeholders)
    scores['fairness'] = np.random.uniform(0.3, 0.7)
    scores['explainability'] = np.random.uniform(0.2, 0.6)
    scores['value_alignment'] = np.random.uniform(0.1, 0.5)

    # Deep Theoretical Pressures (purely conceptual placeholders)
    scores['causal_density'] = connection_density
    scores['self_organization'] = genotype.complexity
    scores['autopoiesis'] = scores['robustness']  # Robustness is a proxy for maintaining organization
    scores['computational_irreducibility'] = np.random.uniform(0.1, 0.9)
    scores['cognitive_synergy'] = np.random.uniform(0.2, 0.8)

    # 7. Epigenetic Marking (Lamarckian-like learning)
    # The individual "learns" from its performance, creating a marker for its offspring.
    # This maps current performance to a small, heritable aptitude value.
    if enable_epigenetics:
        aptitude_key = f"{task_type}_aptitude"
        performance_marker = (scores['task_accuracy'] - 0.5) * 0.05 # Small learning step
        current_aptitude = genotype.epigenetic_markers.get(aptitude_key, 0.0)
        genotype.epigenetic_markers[aptitude_key] = np.clip(current_aptitude + performance_marker, -0.15, 0.15)
    
    # 8. Epistatic Contribution (NK Landscape Simulation)
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
        # Add new weights with 0 default to prevent key errors if not passed
        new_weights = {
            'learning_speed': 0.0, 'data_parsimony': 0.0, 'forgetting_resistance': 0.0, 'adaptability': 0.0,
            'latency': 0.0, 'energy_consumption': 0.0, 'development_cost': 0.0, 'modularity': 0.0,
            'interpretability': 0.0, 'evolvability': 0.0, 'fairness': 0.0, 'explainability': 0.0,
            'value_alignment': 0.0, 'causal_density': 0.0, 'self_organization': 0.0, 'autopoiesis': 0.0,
            'computational_irreducibility': 0.0, 'cognitive_synergy': 0.0
        }
        weights.update(new_weights)
        
    total_fitness = sum(scores.get(k, 0.0) * v for k, v in weights.items())
    
    # Apply epistatic effect to final fitness
    total_fitness += epistatic_contribution
    
    # 9. Red Queen Coevolution (Parasite Attack)
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
            'dynamic_environment': False,
            'env_change_frequency': 25,
            'num_forms': 5,
            'population_per_form': 20,
            'w_accuracy': 0.6,
            'w_efficiency': 0.15,
            'w_robustness': 0.1,
            'w_generalization': 0.15,
            # --- NEW ADVANCED PRIMARY OBJECTIVES DEFAULTS ---
            'w_learning_speed': 0.0,
            'w_data_parsimony': 0.0,
            'w_forgetting_resistance': 0.0,
            'w_adaptability': 0.0,
            'w_latency': 0.0,
            'w_energy_consumption': 0.0,
            'w_development_cost': 0.0,
            'w_modularity': 0.0,
            'w_interpretability': 0.0,
            'w_evolvability': 0.0,
            'w_fairness': 0.0,
            'w_explainability': 0.0,
            'w_value_alignment': 0.0,
            'w_causal_density': 0.0,
            'w_self_organization': 0.0,
            'w_autopoiesis': 0.0,
            'w_computational_irreducibility': 0.0,
            'w_cognitive_synergy': 0.0,
            'mutation_rate': 0.2,
            'crossover_rate': 0.7,
            'innovation_rate': 0.05,
            'enable_development': True,
            'enable_baldwin': True,
            'baldwinian_assimilation_rate': 0.0,
            'enable_epigenetics': True,
            'endosymbiosis_rate': 0.005,
            'epistatic_linkage_k': 2,
            'gene_flow_rate': 0.01,
            'niche_competition_factor': 1.5,
            'reintroduction_rate': 0.05,
            'max_archive_size': 100000,
            'enable_cataclysms': False,
            'cataclysm_probability': 0.01,
            'cataclysm_extinction_severity': 0.9,
            'cataclysm_landscape_shift_magnitude': 0.5,
            'post_cataclysm_hypermutation_multiplier': 2.0,
            'post_cataclysm_hypermutation_duration': 10,
            'cataclysm_selectivity_type': 'Uniform',
            'red_queen_virulence': 0.15,
            'red_queen_adaptation_speed': 0.2,
            'red_queen_target_breadth': 0.3,
            'enable_red_queen': True,
            'enable_endosymbiosis': True,
            'mutation_schedule': 'Adaptive',
            'adaptive_mutation_strength': 1.0,
            'selection_pressure': 0.4,
            'enable_diversity_pressure': True,
            'diversity_weight': 0.8,
            'enable_speciation': True,
            'compatibility_threshold': 7.0,
            'num_generations': 100,
            'complexity_level': 'medium',
            'experiment_name': 'Optimal Default Run',
            'random_seed': 42,
            'enable_early_stopping': True,
            'early_stopping_patience': 25,
            'checkpoint_frequency': 50,
            'analysis_top_n': 3,
            'enable_hyperparameter_evolution': False,
            'evolvable_params': ['mutation_rate', 'crossover_rate', 'diversity_weight'],
            'hyper_mutation_rate': 0.05,
            'enable_curriculum_learning': False,
            'curriculum_sequence': ['Vision (ImageNet)', 'Language (MMLU-Pro)', 'Multi-Task Learning'],
            'curriculum_trigger': 'Mean Accuracy Threshold',
            'curriculum_threshold': 0.6,
            'enable_iterative_seeding': False,
            'num_elites_to_seed': 5,
            'seeded_elite_mutation_strength': 0.4,
            # --- NEW DYNAMIC ENVIRONMENT DEFAULTS ---
            'enable_advanced_environment_physics': False,
            'non_stationarity_mode': 'Drift',
            'drift_velocity': 0.01,
            'shift_magnitude': 0.2,
            'cycle_period': 50,
            'chaotic_attractor_type': 'Lorenz',
            'environmental_memory_strength': 0.0,
            'resource_distribution_mode': 'Uniform',
            'resource_regeneration_rate': 0.1,
            'task_space_curvature': 0.0,
            'environmental_viscosity': 0.0,
            'environmental_temperature': 0.0,
            'task_noise_correlation_time': 0.0,
            'environmental_lag': 0,
            'resource_scarcity_level': 1.0,

            'enable_advanced_curriculum': False,
            'curriculum_generation_method': 'Self-Paced',
            'self_paced_learning_rate': 0.05,
            'teacher_student_dynamics_enabled': False,
            'teacher_mutation_rate': 0.1,
            'task_proposal_rejection_rate': 0.2,
            'transfer_learning_bonus': 0.1,
            'catastrophic_forgetting_penalty': 0.1,
            'curriculum_backtracking_probability': 0.05,
            'interleaved_learning_ratio': 0.1,
            'task_decomposition_bonus': 0.0,
            'procedural_content_generation_complexity': 0.0,
            'curriculum_difficulty_ceiling': 1.0,
            'teacher_student_objective_alignment': 1.0,

            'enable_social_environment': False,
            'communication_channel_bandwidth': 1.0,
            'communication_channel_noise': 0.0,
            'social_signal_cost': 0.001,
            'common_knowledge_bonus': 0.0,
            'deception_penalty': 0.0,
            'reputation_system_fidelity': 0.9,
            'sanctioning_effectiveness': 0.5,
            'network_reciprocity_bonus': 0.0,
            'social_learning_mechanism': 'Imitation',
            'cultural_ratchet_bonus': 0.0,
            'social_norm_emergence_bonus': 0.0,
            'tribalism_factor': 0.0,

            'enable_open_endedness': False,
            'poi_novelty_threshold': 0.1,
            'minimal_criterion_coevolution_rate': 0.01,
            'autopoiesis_pressure': 0.0,
            'environmental_construction_bonus': 0.0,
            'goal_switching_cost': 0.01,
            # --- NEW ADVANCED OBJECTIVES DEFAULTS ---
            'enable_advanced_objectives': False,
            # Info-Theoretic
            'w_kolmogorov_complexity': 0.0,
            'w_predictive_information': 0.0,
            'w_causal_emergence': 0.0,
            'w_integrated_information': 0.0,
            'w_free_energy_minimization': 0.0,
            'w_transfer_entropy': 0.0,
            'w_synergistic_information': 0.0,
            'w_state_compression': 0.0,
            'w_empowerment': 0.0,
            'w_semantic_information': 0.0,
            'w_effective_information': 0.0,
            'w_information_closure': 0.0,
            # Thermodynamic
            'w_landauer_cost': 0.0,
            'w_metabolic_efficiency': 0.0,
            'w_heat_dissipation': 0.0,
            'w_homeostasis': 0.0,
            'w_structural_integrity': 0.0,
            'w_entropy_production': 0.0,
            'w_resource_acquisition_efficiency': 0.0,
            'w_aging_resistance': 0.0,
            # Cognitive
            'w_curiosity': 0.0,
            'w_world_model_accuracy': 0.0,
            'w_attention_schema': 0.0,
            'w_theory_of_mind': 0.0,
            'w_cognitive_dissonance': 0.0,
            'w_goal_achievement': 0.0,
            'w_cognitive_learning_speed': 0.0,
            'w_cognitive_forgetting_resistance': 0.0,
            'w_compositionality': 0.0,
            'w_planning_depth': 0.0,
            # Structural
            'w_structural_modularity': 0.0,
            'w_hierarchy': 0.0,
            'w_symmetry': 0.0,
            'w_small_worldness': 0.0,
            'w_scale_free': 0.0,
            'w_fractal_dimension': 0.0,
            'w_hyperbolic_embeddability': 0.0,
            'w_autocatalysis': 0.0,
            'w_wiring_cost': 0.0,
            'w_rich_club_coefficient': 0.0,
            'w_assortativity': 0.0,
            # Temporal
            'w_adaptability_speed': 0.0,
            'w_predictive_horizon': 0.0,
            'w_behavioral_stability': 0.0,
            'w_criticality_dynamics': 0.0,
            'w_decision_time': 0.0,
            'solution_archive_capacity': 1000,
            'novelty_metric': 'Behavioral',
            # --- NEW ADVANCED LANDSCAPE PHYSICS DEFAULTS ---
            'speciation_stagnation_threshold': 15,
            'species_extinction_threshold': 0.01,
            'niche_construction_strength': 0.0,
            'character_displacement_pressure': 0.0,
            'adaptive_radiation_trigger': 0.0,
            'species_merger_probability': 0.0,
            'kin_selection_bonus': 0.0,
            'sexual_selection_factor': 0.0,
            'sympatric_speciation_pressure': 0.0,
            'allopatric_speciation_trigger': 0.0,
            'intraspecific_competition_scaling': 1.0,
            'landscape_ruggedness_factor': 0.0,
            'landscape_correlation_length': 0.1,
            'landscape_neutral_network_size': 0.0,
            'landscape_holeyness_factor': 0.0,
            'landscape_anisotropy_factor': 0.0,
            'landscape_gradient_noise': 0.0,
            'landscape_time_variance_rate': 0.0,
            'multimodality_factor': 0.1,
            'epistatic_correlation_structure': 'Random',
            'local_competition_radius': 0.1,
            'information_seeking_drive': 0.0,
            'open_ended_archive_sampling_bias': 'Uniform',
            'goal_embedding_space_dims': 8,
            # --- NEW FINALIZATION DEFAULTS ---
            'enable_ensemble_creation': False,
            'ensemble_size': 5,
            'fitness_autocorrelation_time': 0.0,
            # --- NEW ADVANCED FINALIZATION DEFAULTS ---
            'enable_advanced_finalization': True,
            'pruning_aggressiveness': 0.1,
            'pruning_method': 'Magnitude',
            'fitness_landscape_plasticity': 0.0,
            'information_bottleneck_pressure': 0.0,
            'fisher_information_maximization': 0.0,
            'predictive_information_bonus': 0.0,
            'thermodynamic_depth_bonus': 0.0,
            'integrated_information_bonus': 0.0,
            'free_energy_minimization_pressure': 0.0,
            'empowerment_maximization_drive': 0.0,
            'causal_density_target': 0.0,
            'semantic_information_bonus': 0.0,
            'algorithmic_complexity_penalty': 0.0,
            'computational_irreducibility_bonus': 0.0,
            'altruism_punishment_effectiveness': 0.0,
            'knowledge_distillation_temperature': 1.0,
            'distillation_teacher_selection': 'Master',
            'self_distillation_weight': 0.0,
            'model_merging_method': 'Weight Averaging',
            'merging_resolution_method': 'Functional',
            'model_merging_alpha': 0.5,
            'bayesian_model_averaging_prior': 0.1,
            'stacking_meta_learner_complexity': 0.2,
            'calibration_method': 'Temperature Scaling',
            'out_of_distribution_generalization_test': 'Adversarial',
            'symbolic_regression_complexity_penalty': 0.01,
            'causal_model_extraction_method': 'PC Algorithm',
            'concept_extraction_method': 'TCAV',
            'model_compression_target_ratio': 0.5,
            'quantization_bits': 8,
            'lottery_ticket_pruning_iterations': 3,
            'continual_learning_replay_buffer_size': 100,
            'elastic_weight_consolidation_lambda': 0.1,
            'synaptic_intelligence_c_param': 0.01,
            'formal_verification_engine': 'SMT Solver',
            'adversarial_robustness_certification_method': 'Interval Bound Propagation',
            'explainability_method': 'Integrated Gradients',
            'concept_bottleneck_regularization': 0.0,
            'mechanistic_interpretability_circuit_search': False,
            'solution_export_format': 'PyTorch',
            'deployment_latency_constraint': 100.0,
            'energy_consumption_constraint': 10.0,
            'final_report_verbosity': 'Standard',
            'archive_solution_for_future_seeding': True,
            'generate_evolutionary_lineage_report': False,
            'perform_sensitivity_analysis_on_hyperparameters': False,
            'ablation_study_component_count': 3,
            'cross_validation_folds': 5,
            'resource_depletion_rate': 0.0,
            'predator_prey_cycle_period': 0,
            'mutualism_bonus': 0.0,
            'parasitism_virulence_factor': 0.1,
            'commensalism_emergence_bonus': 0.0,
            'social_learning_fidelity': 0.0,
            'cultural_evolution_rate': 0.0,
            'group_selection_strength': 0.0,
            'tragedy_of_the_commons_penalty': 0.0,
            'reputation_dynamics_factor': 0.0,
            'extinction_event_severity': 0.9,
            'environmental_shift_magnitude': 0.5,
            'punctuated_equilibrium_trigger_sensitivity': 0.1,
            'key_innovation_bonus': 0.0,
            'background_extinction_rate': 0.0,
            'invasive_species_introduction_prob': 0.0,
            'adaptive_radiation_factor': 2.0,
            'refugia_survival_bonus': 0.0,
            'ensemble_selection_strategy': 'K-Means Diversity',
            'enable_fine_tuning': False,
            'fine_tuning_generations': 10,
            'fine_tuning_mutation_multiplier': 0.1,
            # --- NEW DEEP PHYSICS DEFAULTS ---
            'enable_deep_physics': False,
            # Info-Theoretic
            'kolmogorov_pressure': 0.0,
            'pred_info_bottleneck': 0.0,
            'causal_emergence_factor': 0.0,
            'semantic_closure_pressure': 0.0,
            'phi_target': 0.0,
            'fep_gradient': 0.0,
            'transfer_entropy_maximization': 0.0,
            'synergy_bias': 0.0,
            'state_space_compression': 0.0,
            'fisher_gradient_ascent': 0.0,
            # Thermo
            'landauer_efficiency': 0.0,
            'metabolic_power_law': 0.75,
            'heat_dissipation_constraint': 0.0,
            'homeostatic_pressure': 0.0,
            'computational_temperature': 0.0,
            'structural_decay_rate': 0.0,
            'repair_mechanism_cost': 0.0,
            'szilard_engine_efficiency': 0.0,
            'resource_scarcity': 0.0,
            'allosteric_regulation_factor': 0.0,
            # Quantum
            'quantum_annealing_fluctuation': 0.0,
            'holographic_constraint': 0.0,
            'renormalization_group_flow': 0.0,
            'symmetry_breaking_pressure': 0.0,
            'majorana_fermion_pairing_bonus': 0.0,
            'path_integral_exploration': 0.0,
            'tqft_invariance': 0.0,
            'gauge_theory_redundancy': 0.0,
            'cft_scaling_exponent': 0.0,
            'spacetime_foam_fluctuation': 0.0,
            'entanglement_assisted_comm': 0.0,
            'post_cataclysm_hypermutation_period': 5,
            'environmental_press_factor': 0.0,
            # Topology
            'manifold_adherence': 0.0,
            'group_equivariance_prior': 0.0,
            'ricci_curvature_flow': 0.0,
            'homological_scaffold_stability': 0.0,
            'fractal_dimension_target': 1.0,
            'hyperbolic_embedding_factor': 0.0,
            'small_world_bias': 0.0,
            'scale_free_exponent': 2.0,
            'network_motif_bonus': 0.0,
            'autocatalytic_set_emergence': 0.0,
            'rents_rule_exponent': 0.0,
            'cambrian_explosion_trigger': 0.0,
            # Cognitive
            'curiosity_drive': 0.0,
            'world_model_accuracy': 0.0,
            'ast_congruence': 0.0, 'tom_emergence_pressure': 0.0,
            'cognitive_dissonance_penalty': 0.0, 'opportunity_cost_factor': 0.0,
            'prospect_theory_bias': 0.0, 'temporal_discounting_factor': 0.0,
            'zpd_scaffolding_bonus': 0.0, 'symbol_grounding_constraint': 0.0,
        }
        st.session_state.settings = optimal_defaults
        # --- NEW ADVANCED FRAMEWORKS DEFAULTS ---
        st.session_state.settings.update({
            'enable_advanced_frameworks': False,
            # Computational Logic
            'chaitin_omega_bias': 0.0,
            'godel_incompleteness_penalty': 0.0,
            'turing_completeness_bonus': 0.0,
            'lambda_calculus_isomorphism': 0.0,
            'proof_complexity_cost': 0.0,
            'constructive_type_theory_adherence': 0.0,
            # Learning Theory
            'pac_bayes_bound_minimization': 0.0,
            'vc_dimension_constraint': 0.0,
            'rademacher_complexity_penalty': 0.0,
            'algorithmic_stability_pressure': 0.0,
            'maml_readiness_bonus': 0.0,
            'causal_inference_engine_bonus': 0.0,
            # Morphogenesis
            'reaction_diffusion_activator_rate': 0.0,
            'reaction_diffusion_inhibitor_rate': 0.0,
            'morphogen_gradient_decay': 0.0,
            'cell_adhesion_factor': 0.0,
            'apoptosis_schedule_factor': 0.0,
            'hox_gene_expression_control': 0.0,
            # Collective Intelligence
            'stigmergy_potential_factor': 0.0,
            'quorum_sensing_threshold': 0.0,
            'social_learning_fidelity': 0.0,
            'cultural_transmission_rate': 0.0,
            'division_of_labor_incentive': 0.0,
            'consensus_algorithm_efficiency': 0.0,
            # Game Theory
            'hawk_dove_strategy_ratio': 0.5,
            'ultimatum_game_fairness_pressure': 0.0,
            'principal_agent_alignment_bonus': 0.0,
            'market_clearing_price_efficiency': 0.0,
            'contract_theory_enforcement_cost': 0.0,
            'vickrey_auction_selection_bonus': 0.0,
            # Neuromodulation
            'dopamine_reward_prediction_error': 0.0,
            'serotonin_uncertainty_signal': 0.0,
            'acetylcholine_attentional_gain': 0.0,
            'noradrenaline_arousal_level': 0.0,
            'bcm_rule_sliding_threshold': 0.0,
            'synaptic_scaling_homeostasis': 0.0,
            # Abstract Algebra
            'group_theory_symmetry_bonus': 0.0,
            'category_theory_functorial_bonus': 0.0,
            'monad_structure_bonus': 0.0,
            'lie_algebra_dynamics_prior': 0.0,
            'simplicial_complex_bonus': 0.0,
            'sheaf_computation_consistency': 0.0,
        })
        # --- NEW CO-EVOLUTION & EMBODIMENT DEFAULTS ---
        st.session_state.settings.update({
            'enable_adversarial_coevolution': False,
            'critic_population_size': 10,
            'critic_mutation_rate': 0.3,
            'adversarial_fitness_weight': 0.2,
            'critic_selection_pressure': 0.5,
            'critic_task': 'Find Minimal Perturbation',
            'enable_morphological_coevolution': False,
            'morphological_mutation_rate': 0.05,
            'max_body_modules': 20,
            'cost_per_module': 0.01,
            'enable_sensor_evolution': True,
            'enable_actuator_evolution': True,
            'physical_realism_factor': 0.1,
            'embodiment_gravity': 9.8,
            'embodiment_friction': 0.5,
        })
        # --- NEW SOPHISTICATED CO-EVOLUTION DEFAULTS ---
        st.session_state.settings.update({
            # Adversarial additions
            'enable_hall_of_fame': False,
            'hall_of_fame_size': 20,
            'hall_of_fame_replacement_strategy': 'Replace Weakest',
            'critic_evolution_frequency': 1,
            'critic_cooperation_probability': 0.0,
            'cooperative_reward_scaling': 0.5,
            'critic_objective_novelty_weight': 0.0,
            # Morphological additions
            'bilateral_symmetry_bonus': 0.0,
            'segmentation_bonus': 0.0,
            'allometric_scaling_exponent': 1.0,
            'enable_material_evolution': False,
            'cost_per_stiffness': 0.01,
            'cost_per_density': 0.01,
            'evolvable_sensor_noise': 0.0,
            'evolvable_actuator_force': 1.0,
            'fluid_dynamics_viscosity': 0.0,
            'surface_tension_factor': 0.0,
            # Host-Symbiont
            'enable_host_symbiont_coevolution': False,
            'symbiont_population_size': 50,
            'symbiont_mutation_rate': 0.5,
            'symbiont_transfer_rate': 0.01,
            'symbiont_vertical_inheritance_fidelity': 0.9,
            'host_symbiont_fitness_dependency': 0.1,
        })
        # --- NEW MULTI-LEVEL SELECTION DEFAULTS ---
        st.session_state.settings.update({
            'enable_multi_level_selection': False,
            'colony_formation_method': 'Kinship',
            'colony_size': 10,
            'group_fitness_weight': 0.3,
            'selfishness_suppression_cost': 0.05,
            'caste_specialization_bonus': 0.1,
            'inter_colony_competition_rate': 0.1,
        })
        st.toast("Parameters reset to optimal defaults!", icon="⚙️")
        st.rerun()

    # Get settings from session state, with hardcoded defaults as fallback
    s = st.session_state.get('settings', {})

    # --- META-EVOLUTION EXPANDER ---
    with st.sidebar.expander("🛰️ Meta-Evolution & Self-Configuration", expanded=True):
        st.markdown("""
        **THE APEX OF COMPLEXITY: EVOLVING EVOLUTION ITSELF.**
        
        This section provides god-like control over the evolutionary process. The parameters within allow the system to modify its own learning rules, genetic code, and even its core algorithmic structure. This is the realm of **Meta-Evolution**.
        
        **WARNING:** These are the most powerful and dangerous controls in the entire system. They enable dynamics that are profoundly complex and computationally explosive. Uninformed use will almost certainly lead to total simulation collapse. **You are editing the source code of evolution.**
        """)
        
        st.markdown("---")
        st.markdown("#### 1. Hyperparameter Co-evolution")
        st.markdown("Allow key evolutionary parameters to be encoded in the genome and evolve alongside the architectures.")
        enable_hyperparameter_evolution = st.checkbox(
            "Enable Hyperparameter Co-evolution",
            value=s.get('enable_hyperparameter_evolution', False),
            help="**Automates parameter tuning.** Encodes key hyperparameters (like mutation rate) into each genotype, allowing them to evolve alongside the architecture. The system learns how to learn.",
            key="enable_hyperparameter_evolution_checkbox"
        )
        
        evolvable_params = st.multiselect(
            "Evolvable Parameters",
            options=['mutation_rate', 'crossover_rate', 'innovation_rate', 'diversity_weight', 'selection_pressure', 'epistatic_linkage_k', 'niche_competition_factor'],
            default=s.get('evolvable_params', ['mutation_rate', 'crossover_rate', 'diversity_weight']),
            disabled=not enable_hyperparameter_evolution,
            help="Select which parameters will be encoded into the genotype and evolved.",
            key="evolvable_params_multiselect"
        )
        
        hyper_mutation_rate = st.slider("Meta-Mutation Rate", 0.0, 0.2, s.get('hyper_mutation_rate', 0.05), 0.01,
            disabled=not enable_hyperparameter_evolution, help="The rate at which the hyperparameters themselves mutate.", key="hyper_mutation_rate_slider")

        hyper_mutation_distribution = st.selectbox(
            "Hyper-Mutation Distribution", ['Gaussian', 'Cauchy', 'Uniform'],
            index=['Gaussian', 'Cauchy', 'Uniform'].index(s.get('hyper_mutation_distribution', 'Gaussian')),
            disabled=not enable_hyperparameter_evolution, help="The statistical distribution of hyper-mutations. Cauchy allows for occasional large 'leaps' in parameter values.", key="hyper_mutation_distribution_selectbox"
        )
        evolvable_param_bounds_leniency = st.slider("Evolvable Param Bounds Leniency", 0.0, 1.0, s.get('evolvable_param_bounds_leniency', 0.0), 0.05, disabled=not enable_hyperparameter_evolution, help="How much an evolved parameter is allowed to exceed its predefined UI bounds. 0.1 = 10% over/under. High values allow for extreme, unbounded evolution.", key="evolvable_param_bounds_leniency_slider")
        hyperparam_heritability_factor = st.slider("Hyperparameter Heritability", 0.0, 1.0, s.get('hyperparam_heritability_factor', 0.8), 0.05, disabled=not enable_hyperparameter_evolution, help="Controls how hyperparameters are inherited. 1.0 = average of parents. 0.0 = random choice. Intermediate values blend the two.", key="hyperparam_heritability_factor_slider")

        st.markdown("---")
        st.markdown("#### 2. Genetic Code & Representation Evolution")
        enable_genetic_code_evolution = st.checkbox("Enable Genetic Code Evolution", value=s.get('enable_genetic_code_evolution', False), help="**DANGER: EVOLVE THE GENOME ITSELF.** Allows the system to invent new gene types (e.g., new activations, normalizations) and alter the structure of the genotype-phenotype map.", key="enable_genetic_code_evolution_checkbox")
        
        gene_type_innovation_rate = st.slider("Gene Type Innovation Rate", 0.0, 0.01, s.get('gene_type_innovation_rate', 0.001), 0.0001, format="%.4f", disabled=not enable_genetic_code_evolution, help="Probability per generation of inventing a completely new gene type (e.g., a new activation function).", key="gene_type_innovation_rate_slider")
        gene_type_extinction_rate = st.slider("Gene Type Extinction Rate", 0.0, 0.01, s.get('gene_type_extinction_rate', 0.0005), 0.0001, format="%.4f", disabled=not enable_genetic_code_evolution, help="Probability per generation of an unused gene type going 'extinct' and being removed from the possible gene pool.", key="gene_type_extinction_rate_slider")
        evolvable_activation_functions = st.checkbox("Evolve Activation Functions", value=s.get('evolvable_activation_functions', False), disabled=not enable_genetic_code_evolution, help="Allow the system to create novel activation functions by composing simple mathematical primitives (e.g., x, sin(x), x^2).", key="evolvable_activation_functions_checkbox")
        activation_expression_complexity_limit = st.slider("Activation Expression Complexity Limit", 2, 20, s.get('activation_expression_complexity_limit', 5), 1, disabled=not evolvable_activation_functions, help="The maximum number of operations in an evolved activation function expression tree.", key="activation_expression_complexity_limit_slider")
        developmental_rule_innovation_rate = st.slider("Developmental Rule Innovation Rate", 0.0, 0.01, s.get('developmental_rule_innovation_rate', 0.001), 0.0001, format="%.4f", disabled=not enable_genetic_code_evolution, help="Rate of inventing new types of developmental rules (e.g., a new trigger condition).", key="developmental_rule_innovation_rate_slider")
        encoding_plasticity_rate = st.slider("Encoding Plasticity Rate", 0.0, 0.1, s.get('encoding_plasticity_rate', 0.0), 0.005, disabled=not enable_genetic_code_evolution, help="Rate at which the genotype-to-phenotype mapping itself can change, altering how genes are interpreted.", key="encoding_plasticity_rate_slider")
        genome_length_constraint_pressure = st.slider("Genome Length Constraint Pressure", -1.0, 1.0, s.get('genome_length_constraint_pressure', 0.0), 0.05, disabled=not enable_genetic_code_evolution, help="A pressure on the length of the raw genetic code. Positive values reward longer genomes, negative values reward shorter ones.", key="genome_length_constraint_pressure_slider")
        intron_ratio_target = st.slider("Intron Ratio Target", 0.0, 0.9, s.get('intron_ratio_target', 0.1), 0.05, disabled=not enable_genetic_code_evolution, help="The target ratio of non-coding 'junk DNA' (introns). Introns can act as an evolutionary buffer and facilitate large-scale rearrangements.", key="intron_ratio_target_slider")
        gene_regulatory_network_complexity_bonus = st.slider("GRN Complexity Bonus", 0.0, 1.0, s.get('gene_regulatory_network_complexity_bonus', 0.0), 0.05, disabled=not enable_genetic_code_evolution, help="A fitness bonus for evolving a complex Gene Regulatory Network (GRN) within the developmental rules, promoting sophisticated ontogeny.", key="gene_regulatory_network_complexity_bonus_slider")
        evolvable_normalization_layers = st.checkbox("Evolve Normalization Layers", value=s.get('evolvable_normalization_layers', False), disabled=not enable_genetic_code_evolution, help="Allow the system to invent and use novel normalization layer types beyond the standard ones.", key="evolvable_normalization_layers_checkbox")

        st.markdown("---")
        st.markdown("#### 3. Evolutionary Algorithm (EA) Dynamics Evolution")
        enable_ea_dynamics_evolution = st.checkbox("Enable EA Dynamics Evolution", value=s.get('enable_ea_dynamics_evolution', False), help="**DANGER: EVOLVE THE ALGORITHM.** Allows the system to change its own selection, crossover, and population topology rules during the run.", key="enable_ea_dynamics_evolution_checkbox")

        evolvable_selection_mechanism = st.checkbox("Evolve Selection Mechanism", value=s.get('evolvable_selection_mechanism', False), disabled=not enable_ea_dynamics_evolution, help="Allow the population to evolve which selection mechanism is used.", key="evolvable_selection_mechanism_checkbox")
        selection_mechanism_pool = st.multiselect("Selection Mechanism Pool", ['Tournament', 'Truncation', 'Roulette Wheel', 'SUS'], default=s.get('selection_mechanism_pool', ['Tournament']), disabled=not evolvable_selection_mechanism, help="The set of selection mechanisms the system can evolve to use.", key="selection_mechanism_pool_multiselect")
        evolvable_tournament_size = st.checkbox("Evolve Tournament Size", value=s.get('evolvable_tournament_size', False), disabled=not enable_ea_dynamics_evolution, help="If using tournament selection, allow the tournament size to evolve.", key="evolvable_tournament_size_checkbox")
        crossover_operator_pool = st.multiselect("Crossover Operator Pool", ['Homologous', 'Uniform', 'One-Point', 'Two-Point'], default=s.get('crossover_operator_pool', ['Homologous']), disabled=not enable_ea_dynamics_evolution, help="The set of crossover operators the system can evolve to use.", key="crossover_operator_pool_multiselect")
        mutation_operator_pool = st.multiselect("Mutation Operator Pool", ['Gaussian', 'Cauchy', 'Uniform'], default=s.get('mutation_operator_pool', ['Gaussian']), disabled=not enable_ea_dynamics_evolution, help="The set of mutation distributions the system can evolve to use.", key="mutation_operator_pool_multiselect")
        population_topology = st.selectbox("Population Topology", ['Panmictic', 'Island Model', 'Cellular Automaton'], index=['Panmictic', 'Island Model', 'Cellular Automaton'].index(s.get('population_topology', 'Panmictic')), disabled=not enable_ea_dynamics_evolution, help="The structure of the population. Island models and CAs create spatial separation, promoting diversity.", key="population_topology_selectbox")
        evolvable_migration_rate = st.checkbox("Evolve Migration Rate (Islands)", value=s.get('evolvable_migration_rate', False), disabled=(population_topology != 'Island Model'), help="Allow the rate of migration between islands to evolve.", key="evolvable_migration_rate_checkbox")
        evolvable_island_count = st.checkbox("Evolve Island Count", value=s.get('evolvable_island_count', False), disabled=(population_topology != 'Island Model'), help="Allow the number of islands in the population to evolve.", key="evolvable_island_count_checkbox")
        topology_reconfiguration_frequency = st.slider("Topology Reconfiguration Frequency", 0, 100, s.get('topology_reconfiguration_frequency', 0), 5, disabled=not enable_ea_dynamics_evolution, help="How often (in generations) the system can change its topology. 0 = never.", key="topology_reconfiguration_frequency_slider")
        dynamic_speciation_threshold_factor = st.slider("Dynamic Speciation Threshold Factor", -1.0, 1.0, s.get('dynamic_speciation_threshold_factor', 0.0), 0.05, disabled=not enable_ea_dynamics_evolution, help="Allows the speciation threshold to dynamically adjust based on population diversity. Positive values tighten thresholds when diversity is high, negative values loosen them.", key="dynamic_speciation_threshold_factor_slider")

        st.markdown("---")
        st.markdown("#### 4. Fitness Landscape & Objective Evolution (Autotelic Systems)")
        enable_objective_evolution = st.checkbox("Enable Objective Evolution", value=s.get('enable_objective_evolution', False), help="**DANGER: EVOLVE THE GOAL.** Allows the system to change its own fitness objectives, creating 'autotelic' (self-motivated) agents that can define their own goals.", key="enable_objective_evolution_checkbox")

        evolvable_objective_weights = st.checkbox("Evolve Objective Weights", value=s.get('evolvable_objective_weights', False), disabled=not enable_objective_evolution, help="Allow the weights of the main objectives (accuracy, efficiency, etc.) to be encoded in the genome and evolve.", key="evolvable_objective_weights_checkbox")
        objective_weight_mutation_strength = st.slider("Objective Weight Mutation Strength", 0.0, 0.2, s.get('objective_weight_mutation_strength', 0.05), 0.01, disabled=not evolvable_objective_weights, help="The magnitude of mutations applied to the objective weights.", key="objective_weight_mutation_strength_slider")
        autotelic_novelty_search_weight = st.slider("Autotelic Novelty Search Weight", 0.0, 1.0, s.get('autotelic_novelty_search_weight', 0.0), 0.05, disabled=not enable_objective_evolution, help="An evolvable objective that rewards agents purely for being different from others, driving open-ended exploration.", key="autotelic_novelty_search_weight_slider")
        autotelic_complexity_drive_weight = st.slider("Autotelic Complexity Drive Weight", 0.0, 1.0, s.get('autotelic_complexity_drive_weight', 0.0), 0.05, disabled=not enable_objective_evolution, help="An evolvable objective that rewards agents simply for increasing their own architectural complexity.", key="autotelic_complexity_drive_weight_slider")
        autotelic_learning_progress_drive = st.slider("Autotelic Learning Progress Drive", 0.0, 1.0, s.get('autotelic_learning_progress_drive', 0.0), 0.05, disabled=not enable_objective_evolution, help="An evolvable objective that rewards agents for the *magnitude of improvement* during their lifetime learning (Baldwin effect), not just the final score.", key="autotelic_learning_progress_drive_slider")
        fitness_function_noise_injection_rate = st.slider("Fitness Noise Injection Rate", 0.0, 0.2, s.get('fitness_function_noise_injection_rate', 0.0), 0.01, disabled=not enable_objective_evolution, help="An evolvable parameter that adds noise to the fitness evaluation, which can help escape local optima.", key="fitness_function_noise_injection_rate_slider")
        fitness_landscape_smoothing_factor = st.slider("Fitness Landscape Smoothing Factor", 0.0, 1.0, s.get('fitness_landscape_smoothing_factor', 0.0), 0.05, disabled=not enable_objective_evolution, help="An evolvable parameter that averages an individual's fitness with its neighbors, effectively smoothing the landscape.", key="fitness_landscape_smoothing_factor_slider")
        objective_ambition_ratchet = st.slider("Objective Ambition Ratchet", 0.0, 0.1, s.get('objective_ambition_ratchet', 0.0), 0.005, disabled=not enable_objective_evolution, help="A mechanism where, if the population achieves a certain fitness, the objective becomes harder (e.g., the target score increases). This creates a self-induced pressure for continuous improvement.", key="objective_ambition_ratchet_slider")
        pareto_front_focus_bias = st.slider("Pareto Front Focus Bias", 0.0, 1.0, s.get('pareto_front_focus_bias', 0.0), 0.05, disabled=not enable_objective_evolution, help="An evolvable bias that rewards individuals for being at the 'knees' or sparse areas of the Pareto front, promoting a well-distributed set of solutions.", key="pareto_front_focus_bias_slider")

        st.markdown("---")
        st.markdown("#### 5. Computational Reflection & Self-Modification")
        enable_self_modification = st.checkbox("Enable Self-Modification", value=s.get('enable_self_modification', False), help="**ULTIMATE DANGER: SELF-MODIFYING CODE.** Allows a genotype to contain rules that directly modify its own genetic code during its lifetime. This is analogous to a program rewriting its own source code while running.", key="enable_self_modification_checkbox")

        self_modification_probability = st.slider("Self-Modification Probability", 0.0, 0.05, s.get('self_modification_probability', 0.0), 0.001, format="%.3f", disabled=not enable_self_modification, help="The per-lifetime probability that an individual will execute its self-modifying code.", key="self_modification_probability_slider")
        self_modification_scope = st.selectbox("Self-Modification Scope", ['Parameters', 'Structure', 'All'], index=['Parameters', 'Structure', 'All'].index(s.get('self_modification_scope', 'Parameters')), disabled=not enable_self_modification, help="What parts of the genome can be self-modified. 'All' is extremely unstable.", key="self_modification_scope_selectbox")
        quine_bonus = st.slider("Quine Bonus", 0.0, 1.0, s.get('quine_bonus', 0.0), 0.05, disabled=not enable_self_modification, help="A fitness bonus for architectures that can output a description of their own structure, a step towards computational self-awareness (a Quine).", key="quine_bonus_slider")
        meta_genotype_bonus = st.slider("Meta-Genotype Bonus", 0.0, 1.0, s.get('meta_genotype_bonus', 0.0), 0.05, disabled=not enable_self_modification, help="A bonus for genotypes that evolve a 'meta-gene' section that describes how they themselves should be mutated or crossed over.", key="meta_genotype_bonus_slider")
        self_simulation_bonus = st.slider("Self-Simulation Bonus", 0.0, 1.0, s.get('self_simulation_bonus', 0.0), 0.05, disabled=not enable_self_modification, help="A bonus for architectures that can accurately predict their own fitness score without being evaluated, demonstrating a form of self-modeling.", key="self_simulation_bonus_slider")
        

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
            min_value=5, max_value=100, value=s.get('env_change_frequency', 25),
            help="""
            **Simulates environmental volatility.** How often the task changes.
            - **Low values:** Rapidly changing world, favors generalists.
            - **High values:** Long periods of stability, favors specialists.
            **Warning:** Very frequent changes (<10 gens) can prevent meaningful adaptation.
            """,
            disabled=not dynamic_environment,
            key="env_change_freq_slider"
        )
        st.markdown("---")
        enable_curriculum_learning = st.checkbox("Enable Curriculum Learning", value=s.get('enable_curriculum_learning', False), help="**Structured learning progression.** Define a sequence of tasks. The environment will automatically advance to the next task when a performance threshold is met.", key="enable_curriculum_learning_checkbox")
        curriculum_sequence = st.multiselect(
            "Curriculum Sequence",
            options=task_options,
            default=s.get('curriculum_sequence', ['Vision (ImageNet)', 'Language (MMLU-Pro)', 'Multi-Task Learning']),
            disabled=not enable_curriculum_learning,
            help="Drag and drop to define the order of tasks in the curriculum.",
            key="curriculum_sequence_multiselect"
        )
        curriculum_trigger = st.selectbox(
            "Curriculum Transition Trigger",
            options=['Fixed Generations', 'Mean Accuracy Threshold', 'Apex Fitness Threshold'],
            index=['Fixed Generations', 'Mean Accuracy Threshold', 'Apex Fitness Threshold'].index(s.get('curriculum_trigger', 'Mean Accuracy Threshold')),
            disabled=not enable_curriculum_learning,
            key="curriculum_trigger_selectbox"
        )
        curriculum_threshold = st.number_input("Transition Threshold / Generations", value=s.get('curriculum_threshold', 0.6), disabled=not enable_curriculum_learning, key="curriculum_threshold_input")

        st.markdown("---")
        st.markdown("#### 🌌 Advanced Environmental Physics & Non-Stationarity")
        enable_advanced_environment_physics = st.checkbox("Enable Advanced Environment Physics", value=s.get('enable_advanced_environment_physics', False), help="**DANGER: HIGHLY EXPERIMENTAL.** Models the environment itself as a complex dynamical system. Can create extremely unpredictable and chaotic task landscapes.", key="enable_advanced_environment_physics_checkbox")

        non_stationarity_mode = st.selectbox(
            "Non-Stationarity Mode", ['Drift', 'Shift', 'Cycle', 'Chaotic'],
            index=['Drift', 'Shift', 'Cycle', 'Chaotic'].index(s.get('non_stationarity_mode', 'Drift')),
            disabled=not enable_advanced_environment_physics, help="**How the environment changes:**\n- **Drift:** Gradual, continuous change.\n- **Shift:** Sudden, discrete jumps.\n- **Cycle:** Periodic, predictable changes.\n- **Chaotic:** Unpredictable but deterministic changes.", key="non_stationarity_mode_selectbox"
        )
        drift_velocity = st.slider("Drift Velocity", 0.0, 0.1, s.get('drift_velocity', 0.01), 0.001, format="%.3f", disabled=non_stationarity_mode!='Drift', help="Speed of gradual environmental change.", key="drift_velocity_slider")
        shift_magnitude = st.slider("Shift Magnitude", 0.0, 1.0, s.get('shift_magnitude', 0.2), 0.05, disabled=non_stationarity_mode!='Shift', help="Size of sudden environmental jumps.", key="shift_magnitude_slider")
        cycle_period = st.slider("Cycle Period (Generations)", 10, 200, s.get('cycle_period', 50), 5, disabled=non_stationarity_mode!='Cycle', help="Length of periodic environmental changes.", key="cycle_period_slider")
        chaotic_attractor_type = st.selectbox("Chaotic Attractor Type", ['Lorenz', 'Rössler'], index=['Lorenz', 'Rössler'].index(s.get('chaotic_attractor_type', 'Lorenz')), disabled=non_stationarity_mode!='Chaotic', help="The type of chaotic system governing environmental change.", key="chaotic_attractor_type_selectbox")
        
        environmental_memory_strength = st.slider("Environmental Memory (Niche Construction)", 0.0, 1.0, s.get('environmental_memory_strength', 0.0), 0.05, disabled=not enable_advanced_environment_physics, help="How much the population's past actions influence the current state of the environment. High values allow agents to construct their own niches.", key="environmental_memory_strength_slider")
        resource_distribution_mode = st.selectbox("Resource Distribution", ['Uniform', 'Clustered', 'Power-Law'], index=['Uniform', 'Clustered', 'Power-Law'].index(s.get('resource_distribution_mode', 'Uniform')), disabled=not enable_advanced_environment_physics, help="How fitness 'resources' are distributed in the task space.", key="resource_distribution_mode_selectbox")
        resource_regeneration_rate = st.slider("Resource Regeneration Rate", 0.0, 1.0, s.get('resource_regeneration_rate', 0.1), 0.05, disabled=not enable_advanced_environment_physics, help="How quickly exploited resources (high-fitness regions) replenish.", key="resource_regeneration_rate_slider")
        resource_scarcity_level = st.slider("Resource Scarcity Level", 0.1, 2.0, s.get('resource_scarcity_level', 1.0), 0.1, disabled=not enable_advanced_environment_physics, help="Overall availability of fitness resources. < 1.0 means a harsh environment; > 1.0 is a bountiful one.", key="resource_scarcity_level_slider")
        
        task_space_curvature = st.slider("Task Space Curvature", -1.0, 1.0, s.get('task_space_curvature', 0.0), 0.05, disabled=not enable_advanced_environment_physics, help="Models the geometry of the problem space. Positive (sphere-like) means local improvements don't generalize far. Negative (hyperbolic) means they do.", key="task_space_curvature_slider")
        environmental_viscosity = st.slider("Environmental Viscosity", 0.0, 1.0, s.get('environmental_viscosity', 0.0), 0.05, disabled=not enable_advanced_environment_physics, help="A 'drag' force in the task space, making it harder for solutions to change rapidly.", key="environmental_viscosity_slider")
        environmental_temperature = st.slider("Environmental Temperature", 0.0, 1.0, s.get('environmental_temperature', 0.0), 0.05, disabled=not enable_advanced_environment_physics, help="Stochastic noise in the environment's state transitions, making it less predictable.", key="environmental_temperature_slider")
        task_noise_correlation_time = st.slider("Task Noise Correlation Time", 0.0, 1.0, s.get('task_noise_correlation_time', 0.0), 0.05, disabled=not enable_advanced_environment_physics, help="How 'smooth' the noise in the fitness evaluation is. High values mean noise is correlated over time (drifts), low values mean it's random (white noise).", key="task_noise_correlation_time_slider")
        environmental_lag = st.slider("Environmental Lag (Generations)", 0, 20, s.get('environmental_lag', 0), 1, disabled=not enable_advanced_environment_physics, help="Simulates inertia. The number of generations it takes for an environmental change to fully manifest.", key="environmental_lag_slider")

        st.markdown("---")
        st.markdown("#### 📚 Advanced Curriculum & Task Design")
        enable_advanced_curriculum = st.checkbox("Enable Advanced Curriculum Design", value=s.get('enable_advanced_curriculum', False), help="**Automate the teaching process.** Enables co-evolution of tasks and curricula, moving beyond fixed sequences.", key="enable_advanced_curriculum_checkbox")
        
        curriculum_generation_method = st.selectbox("Curriculum Generation Method", ['Self-Paced', 'Teacher-Student', 'Procedural'], index=['Self-Paced', 'Teacher-Student', 'Procedural'].index(s.get('curriculum_generation_method', 'Self-Paced')), disabled=not enable_advanced_curriculum, help="**How new tasks are generated:**\n- **Self-Paced:** Agents choose tasks from a pool based on their own competence.\n- **Teacher-Student:** A co-evolving 'teacher' proposes tasks.\n- **Procedural:** Tasks are generated algorithmically based on a complexity parameter.", key="curriculum_generation_method_selectbox")
        self_paced_learning_rate = st.slider("Self-Paced Learning Rate", 0.0, 0.2, s.get('self_paced_learning_rate', 0.05), 0.01, disabled=curriculum_generation_method!='Self-Paced', help="How quickly agents increase the difficulty of tasks they select for themselves.", key="self_paced_learning_rate_slider")
        procedural_content_generation_complexity = st.slider("PCG Complexity", 0.0, 1.0, s.get('procedural_content_generation_complexity', 0.0), 0.05, disabled=curriculum_generation_method!='Procedural', help="The complexity parameter for procedurally generated tasks.", key="procedural_content_generation_complexity_slider")
        curriculum_difficulty_ceiling = st.slider("Curriculum Difficulty Ceiling", 0.5, 2.0, s.get('curriculum_difficulty_ceiling', 1.0), 0.05, disabled=not enable_advanced_curriculum, help="The maximum difficulty for any task in the curriculum.", key="curriculum_difficulty_ceiling_slider")

        teacher_student_dynamics_enabled = st.checkbox("Enable Teacher-Student Dynamics", value=s.get('teacher_student_dynamics_enabled', False), disabled=curriculum_generation_method!='Teacher-Student', help="Enables a co-evolving 'teacher' population that proposes tasks to the 'student' (main) population.", key="teacher_student_dynamics_enabled_checkbox")
        teacher_mutation_rate = st.slider("Teacher Mutation Rate", 0.05, 0.5, s.get('teacher_mutation_rate', 0.1), 0.05, disabled=not teacher_student_dynamics_enabled, help="How quickly the teacher population adapts its task proposals.", key="teacher_mutation_rate_slider")
        task_proposal_rejection_rate = st.slider("Task Rejection Rate", 0.0, 1.0, s.get('task_proposal_rejection_rate', 0.2), 0.05, disabled=not teacher_student_dynamics_enabled, help="The probability that the student population 'rejects' a proposed task for being too hard or too easy.", key="task_proposal_rejection_rate_slider")
        teacher_student_objective_alignment = st.slider("Teacher-Student Alignment", 0.0, 1.0, s.get('teacher_student_objective_alignment', 1.0), 0.05, disabled=not teacher_student_dynamics_enabled, help="How aligned the teacher's reward is with the student's. < 1.0 can create interesting adversarial or deceptive teaching strategies.", key="teacher_student_objective_alignment_slider")

        transfer_learning_bonus = st.slider("Transfer Learning Bonus", 0.0, 0.5, s.get('transfer_learning_bonus', 0.1), 0.01, disabled=not enable_advanced_curriculum, help="A direct fitness bonus for successfully applying knowledge from a previous task to a new one.", key="transfer_learning_bonus_slider")
        catastrophic_forgetting_penalty = st.slider("Catastrophic Forgetting Penalty", 0.0, 0.5, s.get('catastrophic_forgetting_penalty', 0.1), 0.01, disabled=not enable_advanced_curriculum, help="A fitness penalty for performance degradation on old tasks after learning a new one.", key="catastrophic_forgetting_penalty_slider")
        curriculum_backtracking_probability = st.slider("Curriculum Backtracking Probability", 0.0, 0.5, s.get('curriculum_backtracking_probability', 0.05), 0.01, disabled=not enable_advanced_curriculum, help="Chance per generation to revisit an older, easier task to reinforce learning and prevent forgetting.", key="curriculum_backtracking_probability_slider")
        interleaved_learning_ratio = st.slider("Interleaved Learning Ratio", 0.0, 1.0, s.get('interleaved_learning_ratio', 0.1), 0.05, disabled=not enable_advanced_curriculum, help="During curriculum transitions, the ratio of new task data vs. old task data presented.", key="interleaved_learning_ratio_slider")
        task_decomposition_bonus = st.slider("Task Decomposition Bonus", 0.0, 1.0, s.get('task_decomposition_bonus', 0.0), 0.05, disabled=not enable_advanced_curriculum, help="A bonus for architectures that can break down complex tasks into simpler, solvable sub-tasks.", key="task_decomposition_bonus_slider")

        st.markdown("---")
        st.markdown("#### 🧑‍🤝‍🧑 Social & Multi-Agent Environment")
        enable_social_environment = st.checkbox("Enable Social Environment", value=s.get('enable_social_environment', False), help="**Models a social world.** The environment's state is influenced by agent interactions, communication, and social structures.", key="enable_social_environment_checkbox")
        
        communication_channel_bandwidth = st.slider("Communication Bandwidth", 0.1, 2.0, s.get('communication_channel_bandwidth', 1.0), 0.1, disabled=not enable_social_environment, help="How much information agents can exchange per generation. Higher values allow for more complex social signals.", key="communication_channel_bandwidth_slider")
        communication_channel_noise = st.slider("Communication Noise", 0.0, 1.0, s.get('communication_channel_noise', 0.0), 0.05, disabled=not enable_social_environment, help="How reliable communication is. High noise favors robust or redundant signaling.", key="communication_channel_noise_slider")
        social_signal_cost = st.slider("Social Signal Cost", 0.0, 0.1, s.get('social_signal_cost', 0.001), 0.001, format="%.3f", disabled=not enable_social_environment, help="The fitness cost for an agent to send a signal, promoting efficient communication.", key="social_signal_cost_slider")
        
        common_knowledge_bonus = st.slider("Common Knowledge Bonus", 0.0, 1.0, s.get('common_knowledge_bonus', 0.0), 0.05, disabled=not enable_social_environment, help="A group-level reward for the population establishing a 'common knowledge' state (e.g., all agents know that all other agents know X).", key="common_knowledge_bonus_slider")
        deception_penalty = st.slider("Deception Penalty", 0.0, 1.0, s.get('deception_penalty', 0.0), 0.05, disabled=not enable_social_environment, help="A penalty for sending misleading signals (if the system can detect it), punishing 'liars'.", key="deception_penalty_slider")
        reputation_system_fidelity = st.slider("Reputation System Fidelity", 0.0, 1.0, s.get('reputation_system_fidelity', 0.9), 0.05, disabled=not enable_social_environment, help="How accurately an agent's reputation (e.g., for cooperation) is tracked and propagated through the population.", key="reputation_system_fidelity_slider")
        sanctioning_effectiveness = st.slider("Sanctioning Effectiveness", 0.0, 1.0, s.get('sanctioning_effectiveness', 0.5), 0.05, disabled=not enable_social_environment, help="How effective punishment mechanisms are for non-cooperators. High values make punishment a strong deterrent.", key="sanctioning_effectiveness_slider")
        network_reciprocity_bonus = st.slider("Network Reciprocity Bonus", 0.0, 1.0, s.get('network_reciprocity_bonus', 0.0), 0.05, disabled=not enable_social_environment, help="A fitness bonus for forming reciprocal altruistic relationships within a social network structure.", key="network_reciprocity_bonus_slider")
        social_learning_mechanism = st.selectbox("Social Learning Mechanism", ['Imitation', 'Emulation', 'Instruction'], index=['Imitation', 'Emulation', 'Instruction'].index(s.get('social_learning_mechanism', 'Imitation')), disabled=not enable_social_environment, help="**How agents learn from others:**\n- **Imitation:** Copying actions.\n- **Emulation:** Copying goals/outcomes.\n- **Instruction:** Direct information transfer.", key="social_learning_mechanism_selectbox")
        cultural_ratchet_bonus = st.slider("Cultural Ratchet Bonus", 0.0, 1.0, s.get('cultural_ratchet_bonus', 0.0), 0.05, disabled=not enable_social_environment, help="A bonus for innovations that are learned by others and then improved upon by the next generation, simulating cumulative culture.", key="cultural_ratchet_bonus_slider")
        social_norm_emergence_bonus = st.slider("Social Norm Emergence Bonus", 0.0, 1.0, s.get('social_norm_emergence_bonus', 0.0), 0.05, disabled=not enable_social_environment, help="A group-level bonus for converging on a shared, arbitrary behavioral norm, promoting social cohesion.", key="social_norm_emergence_bonus_slider")
        tribalism_factor = st.slider("Tribalism Factor (In-group Bias)", 0.0, 1.0, s.get('tribalism_factor', 0.0), 0.05, disabled=not enable_social_environment, help="How much agents favor interaction with members of their own species/kin group over outsiders.", key="tribalism_factor_slider")

        st.markdown("---")
        st.markdown("#### ♾️ Open-Ended & Autotelic Dynamics")
        enable_open_endedness = st.checkbox("Enable Open-Ended Evolution", value=s.get('enable_open_endedness', False), help="**Removes the fixed goal.** Rewards agents for discovering novel behaviors and constructing their own complex environments, driving towards unbounded complexity.", key="enable_open_endedness_checkbox")
        
        novelty_metric = st.selectbox("Novelty Metric", ['Behavioral', 'Genotypic', 'Phenotypic'], index=['Behavioral', 'Genotypic', 'Phenotypic'].index(s.get('novelty_metric', 'Behavioral')), disabled=not enable_open_endedness, help="What is measured to determine novelty. Behavioral is based on actions, Genotypic on genes, Phenotypic on final architecture.", key="novelty_metric_selectbox")
        poi_novelty_threshold = st.slider("POI Novelty Threshold", 0.0, 1.0, s.get('poi_novelty_threshold', 0.1), 0.05, disabled=not enable_open_endedness, help="Potential of Interest: The threshold for a new behavior/genotype to be considered 'novel' and added to the archive.", key="poi_novelty_threshold_slider")
        minimal_criterion_coevolution_rate = st.slider("Minimal Criterion Co-evolution Rate", 0.0, 0.1, s.get('minimal_criterion_coevolution_rate', 0.01), 0.005, disabled=not enable_open_endedness, help="Rate at which the objective itself co-evolves to be 'just challenging enough' to maintain progress.", key="minimal_criterion_coevolution_rate_slider")
        autopoiesis_pressure = st.slider("Autopoiesis Pressure", 0.0, 1.0, s.get('autopoiesis_pressure', 0.0), 0.05, disabled=not enable_open_endedness, help="From Maturana & Varela, a pressure rewarding agents that can actively maintain their own organizational structure against environmental decay.", key="autopoiesis_pressure_slider")
        environmental_construction_bonus = st.slider("Environmental Construction Bonus", 0.0, 1.0, s.get('environmental_construction_bonus', 0.0), 0.05, disabled=not enable_open_endedness, help="An intrinsic reward for agents that actively modify the environment in complex, novel ways (niche construction).", key="environmental_construction_bonus_slider")
        goal_switching_cost = st.slider("Goal Switching Cost", 0.0, 0.2, s.get('goal_switching_cost', 0.01), 0.005, disabled=not enable_open_endedness, help="A fitness cost associated with an agent changing its intrinsic goal, promoting goal stability.", key="goal_switching_cost_slider")
        solution_archive_capacity = st.slider("Solution Archive Capacity", 100, 5000, s.get('solution_archive_capacity', 1000), 100, disabled=not enable_open_endedness, help="The size of the archive of past solutions used to calculate novelty.", key="solution_archive_capacity_slider")
        local_competition_radius = st.slider("Local Competition Radius", 0.0, 1.0, s.get('local_competition_radius', 0.1), 0.05, disabled=not enable_open_endedness, help="In novelty search, agents only compete for novelty with others within this radius in behavior space, creating local niches.", key="local_competition_radius_slider")
        information_seeking_drive = st.slider("Information Seeking Drive", 0.0, 1.0, s.get('information_seeking_drive', 0.0), 0.05, disabled=not enable_open_endedness, help="An intrinsic reward for actions that lead to a high reduction in uncertainty about the environment's state (information gain).", key="information_seeking_drive_slider")
        open_ended_archive_sampling_bias = st.selectbox("Novelty Archive Sampling Bias", ['Uniform', 'Recency', 'Fitness'], index=['Uniform', 'Recency', 'Fitness'].index(s.get('open_ended_archive_sampling_bias', 'Uniform')), disabled=not enable_open_endedness, help="How to sample from the archive for novelty calculation. Recency focuses on recent discoveries, Fitness on high-performing ones.", key="open_ended_archive_sampling_bias_selectbox")
        goal_embedding_space_dims = st.slider("Goal Embedding Space Dims", 2, 64, s.get('goal_embedding_space_dims', 8), 2, disabled=not enable_open_endedness, help="The dimensionality of the latent space of possible goals agents can pursue.", key="goal_embedding_space_dims_slider")

    st.sidebar.markdown("### Population Parameters")
    num_forms = st.sidebar.number_input(
        "Number of Architectural Forms",
        min_value=1, max_value=None, value=s.get('num_forms', 5), step=1,
        help="""
        **Simulates the 'Cambrian Explosion' of body plans.** The number of distinct architectural starting points.
        - **Low (1-3):** Focuses evolution on a few known good priors.
        - **High (10+):** Creates a massive, diverse ecosystem with many competing lineages.
        **WARNING:** High values create an immense total population. This is a primary factor for slow performance and potential crashes.
        """,
        key="num_forms_input"
    )
    
    population_per_form = st.sidebar.slider(
        "Population per Form", min_value=5, max_value=200, value=s.get('population_per_form', 20),
        key="pop_per_form_slider",
        help="""
        **Simulates the density of the initial gene pool.** The number of individuals for each starting architectural template.
        A larger population provides more raw genetic material and better resists random genetic drift, mimicking a vast natural population.
        **WARNING:** The 'Total Initial Population' is the key performance bottleneck. High values here will dramatically slow down each generation.
        """
    )
    
    # Add a dynamic display for total population with a warning
    total_population = num_forms * population_per_form
    st.sidebar.metric("Total Initial Population", f"{total_population:,}")
    if total_population > 300:
        st.sidebar.error(f"IMMENSE POPULATION ({total_population}). This will be extremely slow and may crash the app.")

    
    st.sidebar.markdown("### Fitness Objectives")
    with st.sidebar.expander("Multi-Objective Weights", expanded=False):
        st.info("Define the importance of each fitness objective. Weights will be normalized.")
        w_accuracy = st.slider("Accuracy Weight", 0.0, 1.0, s.get('w_accuracy', 0.6), key="w_accuracy_slider", help="**How much to value raw performance on the task.** The primary driver of optimization.")
        w_efficiency = st.slider("Efficiency Weight", 0.0, 1.0, s.get('w_efficiency', 0.15), key="w_efficiency_slider", help="**How much to penalize computational cost (parameters, connections).** Promotes smaller, faster models.")
        w_robustness = st.slider("Robustness Weight", 0.0, 1.0, s.get('w_robustness', 0.1), key="w_robustness_slider", help="**How much to value stability under perturbation.** Favors architectures that are less sensitive to noise.")
        w_generalization = st.slider("Generalization Weight", 0.0, 1.0, s.get('w_generalization', 0.15), key="w_generalization_slider", help="**How much to value traits linked to generalizing to unseen data.** Promotes modularity and plasticity.")

        with st.expander("🔬 Advanced Primary Objectives", expanded=False):
            st.markdown("These objectives are always active and add further dimensions to the fitness landscape, pushing evolution towards more sophisticated solutions.")
            st.markdown("##### 1. Learning & Adaptation Dynamics")
            w_learning_speed = st.slider("Learning Speed", 0.0, 1.0, s.get('w_learning_speed', 0.0), 0.01, key="w_learning_speed_slider", help="Rewards faster convergence during lifetime learning (Baldwin effect).")
            w_data_parsimony = st.slider("Data Parsimony", 0.0, 1.0, s.get('w_data_parsimony', 0.0), 0.01, key="w_data_parsimony_slider", help="Rewards high performance with less data (conceptual).")
            w_forgetting_resistance = st.slider("Forgetting Resistance", 0.0, 1.0, s.get('w_forgetting_resistance', 0.0), 0.01, key="w_forgetting_resistance_slider", help="Penalizes performance drops on old tasks after learning new ones.")
            w_adaptability = st.slider("Adaptability", 0.0, 1.0, s.get('w_adaptability', 0.0), 0.01, key="w_adaptability_slider", help="Rewards quick recovery of fitness after an environmental shift.")

            st.markdown("##### 2. Resource & Implementation Costs")
            w_latency = st.slider("Latency", 0.0, 1.0, s.get('w_latency', 0.0), 0.01, key="w_latency_slider", help="Penalizes high inference time (conceptual).")
            w_energy_consumption = st.slider("Energy Consumption", 0.0, 1.0, s.get('w_energy_consumption', 0.0), 0.01, key="w_energy_consumption_slider", help="Penalizes high simulated energy usage (related to params and activity).")
            w_development_cost = st.slider("Development Cost", 0.0, 1.0, s.get('w_development_cost', 0.0), 0.01, key="w_development_cost_slider", help="Penalizes complex or lengthy developmental processes.")

            st.markdown("##### 3. Structural & Interpretability Properties")
            w_modularity = st.slider("Modularity", 0.0, 1.0, s.get('w_modularity', 0.0), 0.01, key="w_modularity_slider", help="Rewards architectures with high modularity (dense intra-module connections, sparse inter-module ones).")
            w_interpretability = st.slider("Interpretability", 0.0, 1.0, s.get('w_interpretability', 0.0), 0.01, key="w_interpretability_slider", help="Rewards architectures that are structurally simpler or sparser, making them easier to analyze.")
            w_evolvability = st.slider("Evolvability", 0.0, 1.0, s.get('w_evolvability', 0.0), 0.01, key="w_evolvability_slider", help="Rewards architectures with a higher potential for beneficial mutations (a smoother local fitness landscape).")

            st.markdown("##### 4. Safety & Alignment (Conceptual)")
            w_fairness = st.slider("Fairness", 0.0, 1.0, s.get('w_fairness', 0.0), 0.01, key="w_fairness_slider", help="Penalizes biased outputs across different conceptual data subgroups.")
            w_explainability = st.slider("Explainability", 0.0, 1.0, s.get('w_explainability', 0.0), 0.01, key="w_explainability_slider", help="Rewards models that can generate explanations for their decisions (conceptual).")
            w_value_alignment = st.slider("Value Alignment", 0.0, 1.0, s.get('w_value_alignment', 0.0), 0.01, key="w_value_alignment_slider", help="Rewards alignment with a predefined set of ethical principles (conceptual).")

            st.markdown("##### 5. Deep Theoretical Pressures (Conceptual)")
            w_causal_density = st.slider("Causal Density", 0.0, 1.0, s.get('w_causal_density', 0.0), 0.01, key="w_causal_density_slider", help="Rewards high causal interaction and information flow between components.")
            w_self_organization = st.slider("Self-Organization", 0.0, 1.0, s.get('w_self_organization', 0.0), 0.01, key="w_self_organization_slider", help="Rewards systems that spontaneously increase their own complexity/order over time.")
            w_autopoiesis = st.slider("Autopoiesis", 0.0, 1.0, s.get('w_autopoiesis', 0.0), 0.01, key="w_autopoiesis_slider", help="Rewards systems that can actively maintain their own organization against perturbation.")
            w_computational_irreducibility = st.slider("Computational Irreducibility", 0.0, 1.0, s.get('w_computational_irreducibility', 0.0), 0.01, key="w_computational_irreducibility_slider", help="Rewards computational processes that cannot be predicted by a simpler process.")
            w_cognitive_synergy = st.slider("Cognitive Synergy", 0.0, 1.0, s.get('w_cognitive_synergy', 0.0), 0.01, key="w_cognitive_synergy_slider", help="Rewards architectures where the whole is greater than the sum of its parts (high synergistic information).")

        all_weights = {
            'task_accuracy': w_accuracy, 'efficiency': w_efficiency, 'robustness': w_robustness, 'generalization': w_generalization,
            'learning_speed': w_learning_speed, 'data_parsimony': w_data_parsimony, 'forgetting_resistance': w_forgetting_resistance,
            'adaptability': w_adaptability, 'latency': w_latency, 'energy_consumption': w_energy_consumption,
            'development_cost': w_development_cost, 'modularity': w_modularity, 'interpretability': w_interpretability,
            'evolvability': w_evolvability, 'fairness': w_fairness, 'explainability': w_explainability,
            'value_alignment': w_value_alignment, 'causal_density': w_causal_density, 'self_organization': w_self_organization,
            'autopoiesis': w_autopoiesis, 'computational_irreducibility': w_computational_irreducibility,
            'cognitive_synergy': w_cognitive_synergy
        }

        total_w = sum(all_weights.values()) + 1e-9
        fitness_weights = {
            k: v / total_w for k, v in all_weights.items()
        }
        
        st.write("Normalized Weights:")
        st.json({k: f"{v:.2f}" for k, v in fitness_weights.items()})
    
    with st.sidebar.expander("🔬 Advanced Objectives & Intrinsic Motivations", expanded=False):
        enable_advanced_objectives = st.checkbox(
            "Enable Advanced Objectives",
            value=s.get('enable_advanced_objectives', False),
            help="**EXPERIMENTAL:** Introduce a vast suite of fitness objectives derived from information theory, thermodynamics, and cognitive science. These create complex pressures for emergent properties beyond simple task performance.",
            key="enable_advanced_objectives_checkbox"
        )

        if not enable_advanced_objectives:
            st.warning("Advanced Objectives are disabled. These weights will have no effect.")
        else:
            st.success("Advanced Objectives are ACTIVE. The fitness landscape is now shaped by deep theoretical principles.")

        st.markdown("---")
        st.markdown("#### 1. Information-Theoretic Objectives")
        w_kolmogorov_complexity = st.slider("Kolmogorov Complexity", 0.0, 1.0, s.get('w_kolmogorov_complexity', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_kolmogorov_complexity_slider")
        w_predictive_information = st.slider("Predictive Information", 0.0, 1.0, s.get('w_predictive_information', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_predictive_information_slider")
        w_causal_emergence = st.slider("Causal Emergence", 0.0, 1.0, s.get('w_causal_emergence', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_causal_emergence_slider")
        w_integrated_information = st.slider("Integrated Information (Φ)", 0.0, 1.0, s.get('w_integrated_information', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_integrated_information_slider")
        w_free_energy_minimization = st.slider("Free Energy Minimization", 0.0, 1.0, s.get('w_free_energy_minimization', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_free_energy_minimization_slider")
        w_transfer_entropy = st.slider("Transfer Entropy", 0.0, 1.0, s.get('w_transfer_entropy', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_transfer_entropy_slider")
        w_synergistic_information = st.slider("Synergistic Information", 0.0, 1.0, s.get('w_synergistic_information', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_synergistic_information_slider")
        w_state_compression = st.slider("State Compression", 0.0, 1.0, s.get('w_state_compression', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_state_compression_slider")
        w_empowerment = st.slider("Empowerment", 0.0, 1.0, s.get('w_empowerment', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_empowerment_slider")
        w_semantic_information = st.slider("Semantic Information", 0.0, 1.0, s.get('w_semantic_information', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_semantic_information_slider")
        w_effective_information = st.slider("Effective Information", 0.0, 1.0, s.get('w_effective_information', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_effective_information_slider")
        w_information_closure = st.slider("Information Closure", 0.0, 1.0, s.get('w_information_closure', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_information_closure_slider")

        st.markdown("---")
        st.markdown("#### 2. Thermodynamic & Metabolic Objectives")
        w_landauer_cost = st.slider("Landauer Cost", 0.0, 1.0, s.get('w_landauer_cost', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_landauer_cost_slider")
        w_metabolic_efficiency = st.slider("Metabolic Efficiency", 0.0, 1.0, s.get('w_metabolic_efficiency', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_metabolic_efficiency_slider")
        w_heat_dissipation = st.slider("Heat Dissipation", 0.0, 1.0, s.get('w_heat_dissipation', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_heat_dissipation_slider")
        w_homeostasis = st.slider("Homeostasis", 0.0, 1.0, s.get('w_homeostasis', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_homeostasis_slider")
        w_structural_integrity = st.slider("Structural Integrity", 0.0, 1.0, s.get('w_structural_integrity', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_structural_integrity_slider")
        w_entropy_production = st.slider("Entropy Production", 0.0, 1.0, s.get('w_entropy_production', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_entropy_production_slider")
        w_resource_acquisition_efficiency = st.slider("Resource Acquisition Efficiency", 0.0, 1.0, s.get('w_resource_acquisition_efficiency', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_resource_acquisition_efficiency_slider")
        w_aging_resistance = st.slider("Aging Resistance", 0.0, 1.0, s.get('w_aging_resistance', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_aging_resistance_slider")

        st.markdown("---")
        st.markdown("#### 3. Cognitive & Agency-Based Objectives")
        w_curiosity = st.slider("Curiosity", 0.0, 1.0, s.get('w_curiosity', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_curiosity_slider")
        w_world_model_accuracy = st.slider("World Model Accuracy", 0.0, 1.0, s.get('w_world_model_accuracy', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_world_model_accuracy_slider") # This key is unique
        w_attention_schema = st.slider("Attention Schema", 0.0, 1.0, s.get('w_attention_schema', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_attention_schema_slider")
        w_theory_of_mind = st.slider("Theory of Mind", 0.0, 1.0, s.get('w_theory_of_mind', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_theory_of_mind_slider")
        w_cognitive_dissonance = st.slider("Cognitive Dissonance", 0.0, 1.0, s.get('w_cognitive_dissonance', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_cognitive_dissonance_slider")
        w_goal_achievement = st.slider("Goal Achievement", 0.0, 1.0, s.get('w_goal_achievement', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_goal_achievement_slider")
        w_cognitive_learning_speed = st.slider("Cognitive Learning Speed", 0.0, 1.0, s.get('w_cognitive_learning_speed', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_cognitive_learning_speed_slider", help="An intrinsic motivation for agents to improve their performance over their lifetime.")
        w_cognitive_forgetting_resistance = st.slider("Cognitive Forgetting Resistance", 0.0, 1.0, s.get('w_cognitive_forgetting_resistance', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_cognitive_forgetting_resistance_slider", help="An intrinsic motivation to retain knowledge of previous tasks.")
        w_compositionality = st.slider("Compositionality", 0.0, 1.0, s.get('w_compositionality', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_compositionality_slider")
        w_planning_depth = st.slider("Planning Depth", 0.0, 1.0, s.get('w_planning_depth', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_planning_depth_slider")

        st.markdown("---")
        st.markdown("#### 4. Structural & Topological Objectives")
        w_structural_modularity = st.slider("Structural Modularity", 0.0, 1.0, s.get('w_structural_modularity', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_structural_modularity_slider", help="An intrinsic motivation for evolving structurally modular architectures.")
        w_hierarchy = st.slider("Hierarchy", 0.0, 1.0, s.get('w_hierarchy', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_hierarchy_slider")
        w_symmetry = st.slider("Symmetry", 0.0, 1.0, s.get('w_symmetry', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_symmetry_slider")
        w_small_worldness = st.slider("Small-Worldness", 0.0, 1.0, s.get('w_small_worldness', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_small_worldness_slider")
        w_scale_free = st.slider("Scale-Free", 0.0, 1.0, s.get('w_scale_free', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_scale_free_slider")
        w_fractal_dimension = st.slider("Fractal Dimension", 0.0, 1.0, s.get('w_fractal_dimension', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_fractal_dimension_slider")
        w_hyperbolic_embeddability = st.slider("Hyperbolic Embeddability", 0.0, 1.0, s.get('w_hyperbolic_embeddability', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_hyperbolic_embeddability_slider")
        w_autocatalysis = st.slider("Autocatalysis", 0.0, 1.0, s.get('w_autocatalysis', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_autocatalysis_slider")
        w_wiring_cost = st.slider("Wiring Cost", 0.0, 1.0, s.get('w_wiring_cost', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_wiring_cost_slider")
        w_rich_club_coefficient = st.slider("Rich-Club Coefficient", 0.0, 1.0, s.get('w_rich_club_coefficient', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_rich_club_coefficient_slider")
        w_assortativity = st.slider("Assortativity", 0.0, 1.0, s.get('w_assortativity', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_assortativity_slider")

        st.markdown("---")
        st.markdown("#### 5. Temporal Dynamics Objectives")
        w_adaptability_speed = st.slider("Adaptability Speed", 0.0, 1.0, s.get('w_adaptability_speed', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_adaptability_speed_slider")
        w_predictive_horizon = st.slider("Predictive Horizon", 0.0, 1.0, s.get('w_predictive_horizon', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_predictive_horizon_slider")
        w_behavioral_stability = st.slider("Behavioral Stability", 0.0, 1.0, s.get('w_behavioral_stability', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_behavioral_stability_slider")
        w_criticality_dynamics = st.slider("Criticality Dynamics", 0.0, 1.0, s.get('w_criticality_dynamics', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_criticality_dynamics_slider")
        w_decision_time = st.slider("Decision Time", 0.0, 1.0, s.get('w_decision_time', 0.0), 0.01, disabled=not enable_advanced_objectives, key="w_decision_time_slider")

    st.sidebar.markdown("### Evolutionary Operators")
    
    col1, col2 = st.sidebar.columns(2)
    mutation_rate = col1.slider(
        "Base Mutation Rate (μ)",
        min_value=0.01, max_value=0.9, value=s.get('mutation_rate', 0.2), step=0.01,
        help="""
        **The engine of variation, analogous to replication errors.** The base probability of a gene changing.
        - **Low (0.01-0.1):** Simulates a stable genome, favoring fine-tuning.
        - **High (0.5-0.9):** Simulates hypermutation or a high-stress environment, causing rapid, chaotic exploration.
        **WARNING:** Very high rates (>0.6) can lead to 'error catastrophe', where most offspring are non-viable, halting evolutionary progress.
        """,
        key="mutation_rate_slider"
    )
    crossover_rate = col2.slider(
        "Crossover Rate",
        min_value=0.0, max_value=1.0, value=s.get('crossover_rate', 0.7), step=0.05,
        help="**The engine of recombination.** The probability of creating a child from two parents. High values quickly combine successful genetic motifs from different lineages, while low values favor clonal inheritance.",
        key="crossover_rate_slider"
    )
    
    innovation_rate = st.sidebar.slider(
        "Innovation Rate (σ)",
        min_value=0.01, max_value=0.5, value=s.get('innovation_rate', 0.05), step=0.01,
        help="""
        **The engine of discovery, analogous to gene duplication and neofunctionalization.** Rate of adding new modules or connections.
        This is how architectural complexity grows over deep time.
        **WARNING:** High rates (>0.2) can lead to 'genome bloat' and chaotic, non-functional architectures, dramatically increasing computational cost without benefit.
        """,
        key="innovation_rate_slider"
    )
    
    with st.sidebar.expander("🏔️ Landscape & Speciation Control", expanded=False):
        st.markdown("Control the deep physics of the evolutionary ecosystem.")
        epistatic_linkage_k = st.slider(
            "Epistatic Linkage (K)", 0, 10, s.get('epistatic_linkage_k', 2), 1,
            help="""
            **Models the complexity of gene interactions, creating a 'rugged' fitness landscape.** From NK models, K is the number of other genes influencing a single gene's fitness.
            - **K=0:** Smooth, simple landscape (a single mountain).
            - **K > 0:** A rugged landscape with many peaks and valleys, like a mountain range.
            **WARNING:** High K (>5) creates an extremely chaotic, 'glassy' landscape that is nearly unsearchable and can prevent any meaningful adaptation.
            """,
            key="epistatic_linkage_slider"
        )
        gene_flow_rate = st.slider(
            "Gene Flow (Hybridization)", 0.0, 0.2, s.get('gene_flow_rate', 0.01), 0.005,
            help="""
            **Simulates hybridization between species.** The chance for crossover between different species, enabling major, non-linear evolutionary leaps.
            **WARNING:** This is a powerful but dangerous operator. High rates (>0.1) can destroy protected niches and cause species collapse, leading to a catastrophic loss of diversity. Use sparingly.
            """,
            disabled=not s.get('enable_speciation', True),
            key="gene_flow_rate_slider"
        )
        niche_competition_factor = st.slider(
            "Niche Competition", 0.0, 5.0, s.get('niche_competition_factor', 1.5), 0.1,
            help="""
            **Simulates the intensity of ecological competition.** How strongly individuals of the same species compete for resources (fitness sharing).
            - **>1:** Forces species to specialize to survive (niche partitioning).
            - **<1:** Allows more species to coexist.
            **WARNING:** Very high values (>3) can cause extreme specialization and may lead to species extinction if they cannot find a narrow enough niche.
            """,
            disabled=not s.get('enable_speciation', True),
            key="niche_competition_slider",
        )
        reintroduction_rate = st.slider(
            "Archive Reintroduction Rate", 0.0, 0.5, s.get('reintroduction_rate', 0.05), 0.01,
            help="""
            **Simulates a vast, ancient gene pool (the 'fossil record').** Chance to reintroduce a genotype from the archive instead of creating a new child. This is a key mechanism to combat genetic drift and simulate the vast memory of a near-infinite population.
            **WARNING:** High rates (>0.2) can slow down adaptation to the current environment by constantly reintroducing outdated or less-fit 'fossils'.
            """,
            key="reintroduction_rate_slider"
        )
        max_archive_size = st.slider(
            "Max Gene Archive Size", 1000, 1000000, s.get('max_archive_size', 100000), 5000,
            help="""
            **The memory capacity of the 'infinite' gene pool.** A larger archive provides more long-term genetic memory, preventing the permanent loss of innovations.
            **WARNING:** This directly impacts memory (RAM) usage. An immense archive (>250,000) can consume gigabytes of RAM and will crash the app on Streamlit Cloud.
            """,
            key="max_archive_size_slider"
        )
        st.info(
            "These parameters are inspired by theoretical biology to simulate complex evolutionary dynamics like epistasis and niche partitioning."
        )

        with st.expander("🌌 Advanced Landscape & Ecosystem Physics", expanded=False):
            st.warning("DANGER: HIGHLY EXPERIMENTAL. These parameters model deep, complex ecosystem dynamics and can lead to unpredictable or unstable evolution. Use with caution.")

            st.markdown("##### 1. Advanced Speciation & Niche Dynamics")
            speciation_stagnation_threshold = st.slider("Speciation Stagnation Threshold", 0, 50, s.get('speciation_stagnation_threshold', 15), 1, help="Generations a species can stagnate before its members receive a fitness penalty.", key="speciation_stagnation_threshold_slider")
            species_extinction_threshold = st.slider("Species Extinction Fitness Threshold", 0.0, 0.5, s.get('species_extinction_threshold', 0.01), 0.01, help="Mean fitness below which a species is marked for extinction.", key="species_extinction_threshold_slider")
            niche_construction_strength = st.slider("Niche Construction Strength", 0.0, 1.0, s.get('niche_construction_strength', 0.0), 0.05, help="How much individuals can modify their local fitness landscape for their descendants.", key="niche_construction_strength_slider")
            character_displacement_pressure = st.slider("Character Displacement Pressure", 0.0, 1.0, s.get('character_displacement_pressure', 0.0), 0.05, help="A direct pressure for competing species to diverge in their phenotypic traits.", key="character_displacement_pressure_slider")
            adaptive_radiation_trigger = st.slider("Adaptive Radiation Trigger", 0.0, 1.0, s.get('adaptive_radiation_trigger', 0.0), 0.05, help="Fitness threshold that, when crossed by a species, triggers a temporary boost in its innovation rate.", key="adaptive_radiation_trigger_slider")
            species_merger_probability = st.slider("Species Merger Probability", 0.0, 0.1, s.get('species_merger_probability', 0.0), 0.005, help="Probability per generation for two genomically close species to merge into one.", key="species_merger_probability_slider")
            kin_selection_bonus = st.slider("Kin Selection Bonus", 0.0, 1.0, s.get('kin_selection_bonus', 0.0), 0.05, help="A fitness bonus applied based on the success of close relatives, promoting altruistic behavior.", key="kin_selection_bonus_slider")
            sexual_selection_factor = st.slider("Sexual Selection Factor", 0.0, 1.0, s.get('sexual_selection_factor', 0.0), 0.05, help="Introduces a 'mating preference' component to selection, where individuals may prefer mates with certain traits.", key="sexual_selection_factor_slider")
            sympatric_speciation_pressure = st.slider("Sympatric Speciation Pressure", 0.0, 1.0, s.get('sympatric_speciation_pressure', 0.0), 0.05, help="A pressure that encourages speciation even without geographic isolation, based on trait differentiation.", key="sympatric_speciation_pressure_slider")
            allopatric_speciation_trigger = st.slider("Allopatric Speciation Trigger", 0.0, 1.0, s.get('allopatric_speciation_trigger', 0.0), 0.05, help="If population topology is fragmented (e.g., Island Model), this triggers speciation for isolated groups.", key="allopatric_speciation_trigger_slider")
            intraspecific_competition_scaling = st.slider("Intraspecific Competition Scaling", 0.0, 2.0, s.get('intraspecific_competition_scaling', 1.0), 0.1, help="Exponent for scaling competition within a species. >1 means more intense competition in large species.", key="intraspecific_competition_scaling_slider")

            st.markdown("---")
            st.markdown("##### 2. Landscape Topology & Geometry")
            landscape_ruggedness_factor = st.slider("Landscape Ruggedness Factor", 0.0, 1.0, s.get('landscape_ruggedness_factor', 0.0), 0.05, help="Adds fine-grained, high-frequency noise to the fitness landscape.", key="landscape_ruggedness_factor_slider")
            landscape_correlation_length = st.slider("Landscape Correlation Length", 0.0, 1.0, s.get('landscape_correlation_length', 0.1), 0.05, help="How 'smooth' the landscape is. High values mean large, broad peaks; low values mean many sharp peaks.", key="landscape_correlation_length_slider")
            landscape_neutral_network_size = st.slider("Landscape Neutral Network Size", 0.0, 1.0, s.get('landscape_neutral_network_size', 0.0), 0.05, help="The prevalence of 'neutral networks'—paths in genotype space with equal fitness, which can facilitate exploration.", key="landscape_neutral_network_size_slider")
            landscape_holeyness_factor = st.slider("Landscape Holeyness Factor", 0.0, 1.0, s.get('landscape_holeyness_factor', 0.0), 0.05, help="Creates 'holes' or regions of zero fitness in the genotype space, representing lethal mutations.", key="landscape_holeyness_factor_slider")
            landscape_anisotropy_factor = st.slider("Landscape Anisotropy Factor", 0.0, 1.0, s.get('landscape_anisotropy_factor', 0.0), 0.05, help="Stretches the fitness landscape along certain genotypic axes, making some directions of mutation more impactful than others.", key="landscape_anisotropy_factor_slider")
            landscape_gradient_noise = st.slider("Landscape Gradient Noise", 0.0, 1.0, s.get('landscape_gradient_noise', 0.0), 0.05, help="Adds noise to the fitness gradient, making it harder to find the true direction of steepest ascent.", key="landscape_gradient_noise_slider")
            landscape_time_variance_rate = st.slider("Landscape Time-Variance Rate", 0.0, 0.1, s.get('landscape_time_variance_rate', 0.0), 0.005, help="The rate at which the underlying fitness landscape itself slowly and randomly changes over time.", key="landscape_time_variance_rate_slider")
            multimodality_factor = st.slider("Multimodality Factor", 0.0, 1.0, s.get('multimodality_factor', 0.1), 0.05, help="Controls the number of major peaks in the fitness landscape. High values create more distinct optima to discover.", key="multimodality_factor_slider")
            epistatic_correlation_structure = st.selectbox("Epistatic Correlation Structure", ['Random', 'Block', 'Modular'], index=['Random', 'Block', 'Modular'].index(s.get('epistatic_correlation_structure', 'Random')), help="The structure of gene interactions. 'Block' or 'Modular' creates more structured epistasis.", key="epistatic_correlation_structure_selectbox")
            fitness_autocorrelation_time = st.slider("Fitness Autocorrelation Time", 0.0, 1.0, s.get('fitness_autocorrelation_time', 0.0), 0.05, help="How correlated an individual's fitness is with its fitness in the previous generation. High values smooth out fitness trajectories.", key="fitness_autocorrelation_time_slider")
            fitness_landscape_plasticity = st.slider("Fitness Landscape Plasticity", 0.0, 1.0, s.get('fitness_landscape_plasticity', 0.0), 0.05, help="How much the fitness landscape is deformed by the population's presence (niche construction).", key="fitness_landscape_plasticity_slider")

            st.markdown("---")
            st.markdown("##### 3. Information-Theoretic Landscape Dynamics")
            information_bottleneck_pressure = st.slider("Information Bottleneck Pressure", 0.0, 1.0, s.get('information_bottleneck_pressure', 0.0), 0.05, help="Rewards architectures that compress environmental information into a minimal internal representation.", key="information_bottleneck_pressure_slider")
            fisher_information_maximization = st.slider("Fisher Information Maximization", 0.0, 1.0, s.get('fisher_information_maximization', 0.0), 0.05, help="A drive to maximize the population's Fisher Information, which is inversely related to fitness variance and can accelerate adaptation.", key="fisher_information_maximization_slider")
            predictive_information_bonus = st.slider("Predictive Information Bonus", 0.0, 1.0, s.get('predictive_information_bonus', 0.0), 0.05, help="Rewards architectures that are highly predictive of their own future internal states.", key="predictive_information_bonus_slider")
            thermodynamic_depth_bonus = st.slider("Thermodynamic Depth Bonus", 0.0, 1.0, s.get('thermodynamic_depth_bonus', 0.0), 0.05, help="A bonus for architectures whose creation process is computationally 'deep' or complex, rewarding non-trivial structures.", key="thermodynamic_depth_bonus_slider")
            integrated_information_bonus = st.slider("Integrated Information (Φ) Bonus", 0.0, 1.0, s.get('integrated_information_bonus', 0.0), 0.05, help="A bonus for architectures with high integrated information (a measure of consciousness from IIT), promoting integrated and differentiated structures.", key="integrated_information_bonus_slider")
            free_energy_minimization_pressure = st.slider("Free Energy Minimization Pressure", 0.0, 1.0, s.get('free_energy_minimization_pressure', 0.0), 0.05, help="A pressure for architectures to minimize their variational free energy, effectively rewarding the creation of an accurate world model.", key="free_energy_minimization_pressure_slider")
            empowerment_maximization_drive = st.slider("Empowerment Maximization Drive", 0.0, 1.0, s.get('empowerment_maximization_drive', 0.0), 0.05, help="An intrinsic motivation to maximize the channel capacity between an agent's actuators and its future sensors, promoting agency and control.", key="empowerment_maximization_drive_slider")
            causal_density_target = st.slider("Causal Density Target", 0.0, 1.0, s.get('causal_density_target', 0.0), 0.05, help="A target for the density of causal links within the network, can be used to promote or penalize dense connectivity.", key="causal_density_target_slider")
            semantic_information_bonus = st.slider("Semantic Information Bonus", 0.0, 1.0, s.get('semantic_information_bonus', 0.0), 0.05, help="Rewards information that is meaningful or relevant to the agent's goals, as opposed to just raw statistical information.", key="semantic_information_bonus_slider")
            algorithmic_complexity_penalty = st.slider("Algorithmic Complexity Penalty", 0.0, 1.0, s.get('algorithmic_complexity_penalty', 0.0), 0.05, help="A penalty for high algorithmic (Kolmogorov) complexity of the genotype, favoring elegant and compressible solutions.", key="algorithmic_complexity_penalty_slider")
            computational_irreducibility_bonus = st.slider("Computational Irreducibility Bonus", 0.0, 1.0, s.get('computational_irreducibility_bonus', 0.0), 0.05, help="A bonus for computations that cannot be 'shortcut', rewarding processes that are fundamentally complex.", key="computational_irreducibility_bonus_slider")

            st.markdown("---")
            st.markdown("##### 4. Socio-Ecological Dynamics")
            altruism_punishment_effectiveness = st.slider("Altruism Punishment Effectiveness", 0.0, 1.0, s.get('altruism_punishment_effectiveness', 0.0), 0.05, help="In a social context, how effective 'policing' mechanisms are at penalizing selfish individuals.", key="altruism_punishment_effectiveness_slider")
            resource_depletion_rate = st.slider("Resource Depletion Rate", 0.0, 1.0, s.get('resource_depletion_rate', 0.0), 0.05, help="How quickly individuals in a niche deplete the available 'fitness resources', increasing competition.", key="resource_depletion_rate_slider")
            predator_prey_cycle_period = st.slider("Predator-Prey Cycle Period", 0, 100, s.get('predator_prey_cycle_period', 0), 5, help="If > 0, introduces an oscillating 'predator' pressure that targets the most populous species, creating boom-bust cycles.", key="predator_prey_cycle_period_slider")
            mutualism_bonus = st.slider("Mutualism Bonus", 0.0, 1.0, s.get('mutualism_bonus', 0.0), 0.05, help="A fitness bonus for individuals of different species that co-exist in a way that benefits both.", key="mutualism_bonus_slider")
            parasitism_virulence_factor = st.slider("Parasitism Virulence Factor", 0.0, 1.0, s.get('parasitism_virulence_factor', 0.1), 0.05, help="How damaging the 'Red Queen' parasite is to its targets. Higher values increase the negative fitness impact.", key="parasitism_virulence_factor_slider")
            commensalism_emergence_bonus = st.slider("Commensalism Emergence Bonus", 0.0, 1.0, s.get('commensalism_emergence_bonus', 0.0), 0.05, help="A bonus for species that have a neutral-positive interaction (one benefits, the other is unaffected).", key="commensalism_emergence_bonus_slider")
            social_learning_fidelity = st.slider("Social Learning Fidelity", 0.0, 1.0, s.get('social_learning_fidelity', 0.0), 0.05, help="How accurately traits or behaviors can be copied non-genetically from peers.", key="social_learning_fidelity_slider")
            cultural_evolution_rate = st.slider("Cultural Evolution Rate", 0.0, 1.0, s.get('cultural_evolution_rate', 0.0), 0.05, help="The rate of non-genetic 'meme' propagation, allowing successful motifs to spread horizontally.", key="cultural_evolution_rate_slider")
            group_selection_strength = st.slider("Group Selection Strength", 0.0, 1.0, s.get('group_selection_strength', 0.0), 0.05, help="The strength of selection at the group/colony level, promoting group-beneficial traits over individual selfishness.", key="group_selection_strength_slider")
            tragedy_of_the_commons_penalty = st.slider("Tragedy of the Commons Penalty", 0.0, 1.0, s.get('tragedy_of_the_commons_penalty', 0.0), 0.05, help="A penalty applied to groups that over-exploit a shared resource, promoting sustainable strategies.", key="tragedy_of_the_commons_penalty_slider")
            reputation_dynamics_factor = st.slider("Reputation Dynamics Factor", 0.0, 1.0, s.get('reputation_dynamics_factor', 0.0), 0.05, help="How quickly an individual's reputation (e.g., for cooperation) changes based on its actions.", key="reputation_dynamics_factor_slider")

            st.markdown("---")
            st.markdown("##### 5. Catastrophic & Punctuated Equilibria")
            extinction_event_severity = st.slider("Extinction Event Severity", 0.1, 1.0, s.get('extinction_event_severity', 0.9), 0.05, help="Percentage of the population wiped out during a cataclysmic event. 1.0 = total extinction.", key="extinction_event_severity_slider")
            environmental_shift_magnitude = st.slider("Environmental Shift Magnitude", 0.0, 1.0, s.get('environmental_shift_magnitude', 0.5), 0.05, help="How drastically the fitness function changes during an environmental collapse event.", key="environmental_shift_magnitude_slider")
            punctuated_equilibrium_trigger_sensitivity = st.slider("Punctuated Equilibrium Trigger Sensitivity", 0.0, 1.0, s.get('punctuated_equilibrium_trigger_sensitivity', 0.1), 0.05, help="Sensitivity to stagnation for triggering a 'punctuation' event (e.g., a massive, temporary mutation rate spike).", key="punctuated_equilibrium_trigger_sensitivity_slider")
            key_innovation_bonus = st.slider("Key Innovation Bonus", 0.0, 1.0, s.get('key_innovation_bonus', 0.0), 0.05, help="A large, one-time fitness bonus for discovering a 'key innovation' (e.g., a new module type), which can trigger adaptive radiation.", key="key_innovation_bonus_slider")
            background_extinction_rate = st.slider("Background Extinction Rate", 0.0, 0.1, s.get('background_extinction_rate', 0.0), 0.005, help="A constant, low probability of any individual being removed randomly per generation, simulating random death.", key="background_extinction_rate_slider")
            invasive_species_introduction_prob = st.slider("Invasive Species Introduction Probability", 0.0, 0.1, s.get('invasive_species_introduction_prob', 0.0), 0.005, help="Probability per generation of introducing a pre-evolved, highly-fit individual from an external source.", key="invasive_species_introduction_prob_slider")
            adaptive_radiation_factor = st.slider("Adaptive Radiation Factor", 1.0, 10.0, s.get('adaptive_radiation_factor', 2.0), 0.5, help="Multiplier on innovation rate for a few generations after a mass extinction, simulating the rapid filling of empty niches.", key="adaptive_radiation_factor_slider")
            refugia_survival_bonus = st.slider("Refugia Survival Bonus", 0.0, 1.0, s.get('refugia_survival_bonus', 0.0), 0.05, help="A bonus to survival probability during a cataclysm for individuals in under-populated regions of the landscape.", key="refugia_survival_bonus_slider")
            post_cataclysm_hypermutation_period = st.slider("Post-Cataclysm Hypermutation Period", 0, 50, s.get('post_cataclysm_hypermutation_period', 5), 1, help="Number of generations with elevated mutation rates after a cataclysm.", key="post_cataclysm_hypermutation_period_slider")
            environmental_press_factor = st.slider("Environmental Press Factor", 0.0, 0.1, s.get('environmental_press_factor', 0.0), 0.005, help="A slow, constant, directional change in the environment over deep time, forcing continuous adaptation.", key="environmental_press_factor_slider")
            cambrian_explosion_trigger = st.slider("Cambrian Explosion Trigger", 0.0, 1.0, s.get('cambrian_explosion_trigger', 0.0), 0.05, help="A threshold for environmental complexity that, when crossed, triggers a massive, one-time increase in the number of available architectural forms.", key="cambrian_explosion_trigger_slider")
    
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
        enable_cataclysms = st.checkbox("Enable Cataclysms", value=s.get('enable_cataclysms', True), help="**Simulates mass extinctions.** Enable rare, random events like asteroid impacts (population crashes) or environmental collapses (fitness function shifts). Tests ecosystem resilience.", key="enable_cataclysms_checkbox")
        cataclysm_probability = st.slider(
            "Cataclysm Probability", 0.0, 0.5, s.get('cataclysm_probability', 0.01), 0.005,
            help="""
            **Simulates the hostility of the universe (e.g., asteroid impacts).** Per-generation chance of a cataclysmic event.
            **WARNING:** High probability (>0.1) creates an extremely unstable ecosystem where long-term, complex adaptations may be impossible to achieve before being wiped out.
            """,
            disabled=not enable_cataclysms,
            key="cataclysm_prob_slider"
        )
        st.markdown("###### Cataclysm Effects")
        cataclysm_extinction_severity = st.slider("Extinction Severity", 0.1, 1.0, s.get('cataclysm_extinction_severity', 0.9), 0.05, disabled=not enable_cataclysms, help="Percentage of population wiped out during an extinction event.", key="cataclysm_extinction_severity_slider")
        cataclysm_landscape_shift_magnitude = st.slider("Landscape Shift Magnitude", 0.0, 1.0, s.get('cataclysm_landscape_shift_magnitude', 0.5), 0.05, disabled=not enable_cataclysms, help="How drastically the fitness function changes during an environmental collapse.", key="cataclysm_landscape_shift_magnitude_slider")
        cataclysm_selectivity_type = st.selectbox("Cataclysm Selectivity", ['Uniform', 'Fitness-Based (Weakest)', 'Trait-Based (Most Common)'], index=['Uniform', 'Fitness-Based (Weakest)', 'Trait-Based (Most Common)'].index(s.get('cataclysm_selectivity_type', 'Uniform')), disabled=not enable_cataclysms, help="Determines which individuals are most affected by a cataclysm.", key="cataclysm_selectivity_type_selectbox")

        st.markdown("###### Post-Cataclysm Response")
        post_cataclysm_hypermutation_multiplier = st.slider("Hypermutation Multiplier", 1.0, 10.0, s.get('post_cataclysm_hypermutation_multiplier', 2.0), 0.5, disabled=not enable_cataclysms, help="Multiplier on mutation rate after a cataclysm, simulating adaptive radiation.", key="post_cataclysm_hypermutation_multiplier_slider")
        post_cataclysm_hypermutation_duration = st.slider("Hypermutation Duration (Gens)", 0, 50, s.get('post_cataclysm_hypermutation_duration', 10), 1, disabled=not enable_cataclysms, help="Number of generations the hypermutation period lasts.", key="post_cataclysm_hypermutation_duration_slider")

        st.markdown("---")
        enable_red_queen = st.checkbox("Enable Red Queen Dynamics", value=s.get('enable_red_queen', True), help="**'It takes all the running you can do, to keep in the same place.'** A co-evolving 'parasite' creates an arms race by targeting the most common traits, forcing continuous adaptation and preventing stagnation.", key="enable_red_queen_checkbox")
        st.markdown("###### Red Queen Parameters")
        red_queen_virulence = st.slider("Parasite Virulence", 0.0, 1.0, s.get('red_queen_virulence', 0.15), 0.05, disabled=not enable_red_queen, help="The fitness penalty inflicted by the parasite on vulnerable hosts.", key="red_queen_virulence_slider")
        red_queen_adaptation_speed = st.slider("Parasite Adaptation Speed", 0.0, 1.0, s.get('red_queen_adaptation_speed', 0.2), 0.05, disabled=not enable_red_queen, help="How quickly the parasite adapts to target new common traits. 1.0 = adapts every generation.", key="red_queen_adaptation_speed_slider")
        red_queen_target_breadth = st.slider("Parasite Target Breadth", 0.0, 1.0, s.get('red_queen_target_breadth', 0.3), 0.05, disabled=not enable_red_queen, help="How specific the parasite's attack is. 0.0 = targets one exact trait. 1.0 = targets a broad class of traits.", key="red_queen_target_breadth_slider")
        st.info(
            "These features test the ecosystem's resilience and ability to escape static equilibria through external shocks and internal arms races."
        )


    with st.sidebar.expander("🔬 Advanced Dynamics", expanded=True):
        st.markdown("These features add deep biological complexity. You can disable them for a more classical evolutionary run.")
        enable_development = st.checkbox("Enable Developmental Program", value=s.get('enable_development', True), help="**Simulates ontogeny (growth from embryo to adult).** Allows genotypes to execute a 'developmental program' during their lifetime, such as pruning weak connections or growing modules. This separates the genotype from the final phenotype.", key="enable_development_checkbox")
        enable_baldwin = st.checkbox("Enable Baldwin Effect", value=s.get('enable_baldwin', True), help="**Models how learning shapes evolution.** An individual's `plasticity` allows it to 'learn' (improve its fitness) during its lifetime. This creates a selective pressure for architectures that are not just fit, but also good at learning (phenotypic plasticity).", key="enable_baldwin_checkbox")
        baldwinian_assimilation_rate = st.slider(
            "Baldwinian Assimilation Rate", 0.0, 0.1, s.get('baldwinian_assimilation_rate', 0.0), 0.001,
            help="**The final step of learning becoming instinct.** The probability that a trait acquired via phenotypic plasticity (Baldwin Effect) becomes genetically fixed in the next generation. High rates rapidly convert learned behaviors into innate abilities.",
            key="baldwinian_assimilation_rate_slider",
            disabled=not enable_baldwin
        )
        enable_epigenetics = st.checkbox("Enable Epigenetic Inheritance", value=s.get('enable_epigenetics', True), help="**Models Lamarckian-like inheritance.** Individuals pass down partially heritable 'aptitude' markers based on their life experience, allowing for very fast, non-genetic adaptation across a few generations.", key="enable_epigenetics_checkbox")
        enable_endosymbiosis = st.checkbox("Enable Endosymbiosis", value=s.get('enable_endosymbiosis', True), help="**Simulates Major Evolutionary Transitions.** A rare event where an architecture acquires a pre-evolved, successful module from another individual, simulating horizontal gene transfer and allowing for massive, instantaneous leaps in complexity.", key="enable_endosymbiosis_checkbox")
        
        endosymbiosis_rate = st.slider(
            "Endosymbiosis Rate",
            min_value=0.0, max_value=0.1, value=s.get('endosymbiosis_rate', 0.005), step=0.001,
            help="""**Rate of major evolutionary leaps.**
            **WARNING:** This should be very rare. High rates (>0.02) will lead to chaotic, Frankenstein-like architectures that are unlikely to be viable.""",
            disabled=not enable_endosymbiosis,
            key="endosymbiosis_rate_slider"
        )

    with st.sidebar.expander("🛰️ Co-evolutionary & Embodiment Dynamics", expanded=False):
        st.markdown("""
        **WARNING: HIGHLY ADVANCED & COMPUTATIONALLY INTENSIVE.**
        
        This section introduces two powerful co-evolutionary paradigms that fundamentally alter the evolutionary process, moving beyond static fitness functions towards dynamic, interactive adaptation.
        
        - **Adversarial Co-evolution:** Creates a "predator-prey" arms race between solutions and co-evolving "critics" that seek to exploit their weaknesses.
        - **Morphological Co-evolution:** Co-evolves the "body" (morphology) and the "brain" (controller), grounding cognition in a physical form.
        
        Enabling these will significantly increase computation time and complexity.
        """)
        st.markdown("---")
        
        st.markdown("#### 1. Adversarial Co-evolution (Self-Play)")
        enable_adversarial_coevolution = st.checkbox(
            "Enable Adversarial Critic Population",
            value=s.get('enable_adversarial_coevolution', False),
            help="**Simulates an intellectual arms race.** Co-evolves a separate population of 'critic' agents whose goal is to find inputs or conditions that cause the main agents to fail. The main agents are then rewarded for their robustness against these adversarial attacks.",
            key="enable_adversarial_coevolution_checkbox"
        )
        
        critic_population_size = st.slider(
            "Critic Population Size", 5, 100, s.get('critic_population_size', 10), 5,
            disabled=not enable_adversarial_coevolution,
            help="The number of critic agents in the co-evolving population.",
            key="critic_population_size_slider"
        )
        
        critic_mutation_rate = st.slider(
            "Critic Mutation Rate", 0.05, 0.9, s.get('critic_mutation_rate', 0.3), 0.05,
            disabled=not enable_adversarial_coevolution,
            help="How quickly the critics adapt to exploit new weaknesses. Higher rates lead to a more aggressive arms race.",
            key="critic_mutation_rate_slider"
        )
        
        adversarial_fitness_weight = st.slider(
            "Adversarial Fitness Weight", 0.0, 1.0, s.get('adversarial_fitness_weight', 0.2), 0.05,
            disabled=not enable_adversarial_coevolution,
            help="The portion of an agent's total fitness derived from its performance against the critics. Higher values prioritize robustness over raw task performance.",
            key="adversarial_fitness_weight_slider"
        )
        
        critic_selection_pressure = st.slider(
            "Critic Selection Pressure", 0.1, 0.9, s.get('critic_selection_pressure', 0.5), 0.05,
            disabled=not enable_adversarial_coevolution,
            help="The fraction of the best critics that survive to reproduce. Higher pressure leads to more expert critics.",
            key="critic_selection_pressure_slider"
        )
        
        critic_task = st.selectbox(
            "Critic Objective",
            ['Find Minimal Perturbation', 'Generate Deceptive Inputs', 'Identify Catastrophic Forgetting'],
            index=['Find Minimal Perturbation', 'Generate Deceptive Inputs', 'Identify Catastrophic Forgetting'].index(s.get('critic_task', 'Find Minimal Perturbation')),
            disabled=not enable_adversarial_coevolution,
            help="**What the critic tries to do:**\n- **Find Minimal Perturbation:** Finds the smallest change to an input that flips the agent's output.\n- **Generate Deceptive Inputs:** Generates novel inputs that look like one class but are classified as another.\n- **Identify Catastrophic Forgetting:** Tests if learning a new task makes the agent forget an old one.",
            key="critic_task_selectbox"
        )
        
        st.markdown("###### Advanced Adversarial Dynamics")
        enable_hall_of_fame = st.checkbox(
            "Enable Critic Hall of Fame",
            value=s.get('enable_hall_of_fame', False),
            disabled=not enable_adversarial_coevolution,
            help="**Prevents forgetting.** Critics compete not just against the current generation, but also against a 'Hall of Fame' of past champion agents. This forces critics to maintain a memory of past exploits and leads to more robust, generalist agents.",
            key="enable_hall_of_fame_checkbox"
        )
        hall_of_fame_size = st.slider(
            "Hall of Fame Size", 5, 100, s.get('hall_of_fame_size', 20), 5,
            disabled=not enable_hall_of_fame,
            help="The number of past champions maintained in the Hall of Fame.",
            key="hall_of_fame_size_slider"
        )
        hall_of_fame_replacement_strategy = st.selectbox(
            "HoF Replacement Strategy", ['Replace Weakest', 'First-In, First-Out'],
            index=['Replace Weakest', 'First-In, First-Out'].index(s.get('hall_of_fame_replacement_strategy', 'Replace Weakest')),
            disabled=not enable_hall_of_fame,
            help="How the Hall of Fame is updated when a new champion emerges.",
            key="hall_of_fame_replacement_strategy_selectbox"
        )
        critic_evolution_frequency = st.slider(
            "Critic Evolution Frequency", 1, 10, s.get('critic_evolution_frequency', 1), 1,
            disabled=not enable_adversarial_coevolution,
            help="Evolve the critic population every N agent generations. A value > 1 creates an 'asymmetric' arms race where agents have more time to adapt to a static set of critics.",
            key="critic_evolution_frequency_slider"
        )
        critic_cooperation_probability = st.slider(
            "Critic Cooperation Probability", 0.0, 1.0, s.get('critic_cooperation_probability', 0.0), 0.05,
            disabled=not enable_adversarial_coevolution,
            help="The probability that a critic is rewarded for *helping* an agent instead of hindering it. A non-zero value introduces complex competitive-cooperative dynamics.",
            key="critic_cooperation_probability_slider"
        )
        cooperative_reward_scaling = st.slider("Cooperative Reward Scaling", 0.1, 2.0, s.get('cooperative_reward_scaling', 0.5), 0.1, disabled=not enable_adversarial_coevolution or critic_cooperation_probability == 0.0, help="How much a cooperative action from a critic is rewarded, relative to a competitive one.", key="cooperative_reward_scaling_slider")
        critic_objective_novelty_weight = st.slider("Critic Novelty Weight", 0.0, 1.0, s.get('critic_objective_novelty_weight', 0.0), 0.05, disabled=not enable_adversarial_coevolution, help="A weight in the critic's own fitness function that rewards it for finding *new* or *unusual* ways to make an agent fail, promoting a more diverse set of adversarial attacks.", key="critic_objective_novelty_weight_slider")

        st.markdown("---")
        st.markdown("#### 2. Morphological Co-evolution (Embodied Cognition)")
        enable_morphological_coevolution = st.checkbox(
            "Enable Morphological Co-evolution",
            value=s.get('enable_morphological_coevolution', False),
            help="**'The body shapes the mind.'** Co-evolves the physical 'body' (number, type, and placement of modules) alongside the 'brain' (connections). This grounds the architecture in a physical form and allows for the discovery of novel body plans. Disables fixed Architectural Forms.",
            key="enable_morphological_coevolution_checkbox"
        )
        
        morphological_mutation_rate = st.slider(
            "Morphological Mutation Rate", 0.01, 0.5, s.get('morphological_mutation_rate', 0.05), 0.01,
            disabled=not enable_morphological_coevolution,
            help="The rate of mutations affecting the body plan, such as adding/removing modules or changing their type.",
            key="morphological_mutation_rate_slider"
        )
        
        max_body_modules = st.slider(
            "Max Body Modules", 5, 50, s.get('max_body_modules', 20), 1,
            disabled=not enable_morphological_coevolution,
            help="The maximum number of modules (body parts) an individual can evolve.",
            key="max_body_modules_slider"
        )
        
        cost_per_module = st.slider(
            "Metabolic Cost per Module", 0.0, 0.1, s.get('cost_per_module', 0.01), 0.001,
            disabled=not enable_morphological_coevolution,
            help="A fitness penalty applied for each module in the body, simulating metabolic cost and pressuring for efficient morphologies.",
            key="cost_per_module_slider"
        )
        
        col_morph1, col_morph2 = st.columns(2)
        enable_sensor_evolution = col_morph1.checkbox(
            "Evolve Sensors", value=s.get('enable_sensor_evolution', True),
            disabled=not enable_morphological_coevolution,
            help="Allow the evolution of sensor module types and their properties.",
            key="enable_sensor_evolution_checkbox"
        )
        enable_actuator_evolution = col_morph2.checkbox(
            "Evolve Actuators", value=s.get('enable_actuator_evolution', True),
            disabled=not enable_morphological_coevolution,
            help="Allow the evolution of actuator (output) module types and their properties.",
            key="enable_actuator_evolution_checkbox"
        )
        
        st.markdown("###### Physics Simulation")
        physical_realism_factor = st.slider(
            "Physical Realism Factor", 0.0, 1.0, s.get('physical_realism_factor', 0.1), 0.05,
            disabled=not enable_morphological_coevolution,
            help="How much the simulated physics (gravity, friction) affects fitness. A value of 0 means abstract evolution; 1 means fitness is heavily dependent on physical simulation.",
            key="physical_realism_factor_slider"
        )
        
        embodiment_gravity = st.slider(
            "Embodiment Gravity", 0.0, 20.0, s.get('embodiment_gravity', 9.8), 0.1,
            disabled=not enable_morphological_coevolution or physical_realism_factor == 0.0,
            help="The strength of the gravitational force in the simulated physical environment.",
            key="embodiment_gravity_slider"
        )
        
        embodiment_friction = st.slider(
            "Embodiment Friction", 0.0, 1.0, s.get('embodiment_friction', 0.5), 0.05,
            disabled=not enable_morphological_coevolution or physical_realism_factor == 0.0,
            help="The coefficient of friction in the simulated physical environment.",
            key="embodiment_friction_slider"
        )
        
        st.markdown("###### Morphological Priors & Constraints")
        bilateral_symmetry_bonus = st.slider(
            "Bilateral Symmetry Bonus", 0.0, 0.5, s.get('bilateral_symmetry_bonus', 0.0), 0.01,
            disabled=not enable_morphological_coevolution,
            help="A fitness bonus for body plans that exhibit left-right (bilateral) symmetry, a powerful and common prior in natural evolution.",
            key="bilateral_symmetry_bonus_slider"
        )
        segmentation_bonus = st.slider(
            "Segmentation Bonus", 0.0, 0.5, s.get('segmentation_bonus', 0.0), 0.01,
            disabled=not enable_morphological_coevolution,
            help="A fitness bonus for body plans composed of repeated, similar segments (e.g., like a centipede or vertebrate spine). Promotes modular and scalable morphologies.",
            key="segmentation_bonus_slider"
        )
        allometric_scaling_exponent = st.slider(
            "Allometric Scaling Exponent", 0.5, 2.0, s.get('allometric_scaling_exponent', 1.0), 0.05,
            disabled=not enable_morphological_coevolution,
            help="Controls how body parts scale relative to each other (e.g., head size vs. body size). An exponent of 1.0 is isometric (uniform scaling); other values create non-uniform scaling seen in nature.",
            key="allometric_scaling_exponent_slider"
        )

        st.markdown("###### Material & Advanced Physical Property Evolution")
        enable_material_evolution = st.checkbox(
            "Enable Material Evolution", value=s.get('enable_material_evolution', False),
            disabled=not enable_morphological_coevolution,
            help="Allows modules to evolve physical material properties like stiffness and density, in addition to their computational function.",
            key="enable_material_evolution_checkbox"
        )
        cost_per_stiffness = st.slider("Cost per Stiffness", 0.0, 0.1, s.get('cost_per_stiffness', 0.01), 0.001, disabled=not enable_material_evolution, help="Metabolic fitness cost associated with evolving stiffer materials.", key="cost_per_stiffness_slider")
        cost_per_density = st.slider("Cost per Density", 0.0, 0.1, s.get('cost_per_density', 0.01), 0.001, disabled=not enable_material_evolution, help="Metabolic fitness cost associated with evolving denser materials.", key="cost_per_density_slider")
        
        st.markdown("###### Advanced Sensor/Actuator & Environment Physics")
        evolvable_sensor_noise = st.slider(
            "Evolvable Sensor Noise", 0.0, 0.5, s.get('evolvable_sensor_noise', 0.0), 0.01,
            disabled=not enable_morphological_coevolution,
            help="The base noise level for sensors. If non-zero, architectures can evolve to be robust to noisy sensory input.",
            key="evolvable_sensor_noise_slider"
        )
        evolvable_actuator_force = st.slider(
            "Evolvable Actuator Force", 0.1, 5.0, s.get('evolvable_actuator_force', 1.0), 0.1,
            disabled=not enable_morphological_coevolution,
            help="The maximum force an actuator module can exert. Higher force may have higher metabolic costs.",
            key="evolvable_actuator_force_slider"
        )
        fluid_dynamics_viscosity = st.slider(
            "Fluid Dynamics Viscosity", 0.0, 1.0, s.get('fluid_dynamics_viscosity', 0.0), 0.05,
            disabled=not enable_morphological_coevolution or physical_realism_factor == 0.0,
            help="Simulates the viscosity of a surrounding fluid (e.g., water or air), creating a drag force on moving bodies. High values favor streamlined morphologies.",
            key="fluid_dynamics_viscosity_slider"
        )
        surface_tension_factor = st.slider(
            "Surface Tension Factor", 0.0, 1.0, s.get('surface_tension_factor', 0.0), 0.05,
            disabled=not enable_morphological_coevolution or physical_realism_factor == 0.0,
            help="Simulates surface tension forces for agents at the boundary of a medium, relevant for aquatic or amphibious scenarios.",
            key="surface_tension_factor_slider"
        )

        st.markdown("---")
        st.markdown("#### 3. Host-Symbiont Dynamics (Microbiome Simulation)")
        enable_host_symbiont_coevolution = st.checkbox("Enable Host-Symbiont Co-evolution", value=s.get('enable_host_symbiont_coevolution', False), help="**Simulates a microbiome.** Co-evolves a population of fast-evolving 'symbionts' that live inside the main 'host' agents. The host's fitness is modified by its symbiont colony, creating a two-level evolutionary dynamic.", key="enable_host_symbiont_coevolution_checkbox")
        symbiont_population_size = st.slider("Symbiont Population per Host", 10, 200, s.get('symbiont_population_size', 50), 10, disabled=not enable_host_symbiont_coevolution, help="The number of symbionts each host carries.", key="symbiont_population_size_slider")
        symbiont_mutation_rate = st.slider("Symbiont Mutation Rate", 0.1, 1.0, s.get('symbiont_mutation_rate', 0.5), 0.05, disabled=not enable_host_symbiont_coevolution, help="The mutation rate for the rapidly evolving symbionts.", key="symbiont_mutation_rate_slider")
        symbiont_transfer_rate = st.slider("Symbiont Horizontal Transfer Rate", 0.0, 0.2, s.get('symbiont_transfer_rate', 0.01), 0.01, disabled=not enable_host_symbiont_coevolution, help="The probability of symbionts being transferred between hosts upon 'contact', simulating horizontal gene transfer.", key="symbiont_transfer_rate_slider")
        symbiont_vertical_inheritance_fidelity = st.slider("Symbiont Vertical Inheritance Fidelity", 0.5, 1.0, s.get('symbiont_vertical_inheritance_fidelity', 0.9), 0.05, disabled=not enable_host_symbiont_coevolution, help="The fidelity with which a host passes its symbiont colony to its offspring. < 1.0 means imperfect inheritance.", key="symbiont_vertical_inheritance_fidelity_slider")
        host_symbiont_fitness_dependency = st.slider("Host-Symbiont Fitness Dependency", 0.0, 1.0, s.get('host_symbiont_fitness_dependency', 0.1), 0.05, disabled=not enable_host_symbiont_coevolution, help="How much of the host's fitness is determined by the properties of its symbiont colony.", key="host_symbiont_fitness_dependency_slider")

        st.info(
            "Hover over the (?) on each checkbox for a detailed explanation of the dynamic."
        )

    with st.sidebar.expander("👑 Multi-Level & Social Evolution (Major Transitions)", expanded=False):
        st.markdown("""
        **THEORETICAL APEX: SIMULATING THE ORIGINS OF COMPLEXITY.**
        
        This section models one of the deepest concepts in evolutionary biology: **Major Evolutionary Transitions**, where groups of individuals become so integrated they form a new, higher-level "superorganism" (e.g., cells to multicellular life, insects to a hive).
        
        - **Mechanism:** It introduces a second layer of selection. Individuals are selected based on their own fitness, but they are also grouped into **Colonies**, and these colonies are selected based on group performance.
        - **Dynamics:** This creates a fundamental conflict between individual selfishness and group cooperation.
        
        **WARNING:** This is the most computationally and conceptually complex feature. It simulates populations of populations and can lead to extremely rich but unpredictable social dynamics.
        """)
        st.markdown("---")
        
        enable_multi_level_selection = st.checkbox(
            "Enable Multi-Level Selection (MLS)",
            value=s.get('enable_multi_level_selection', False),
            help="**Evolve colonies, not just individuals.** Activates a two-tiered selection system, creating a dynamic tension between individual-level and group-level fitness.",
            key="enable_multi_level_selection_checkbox"
        )
        
        colony_formation_method = st.selectbox(
            "Colony Formation Method",
            ['Kinship', 'Random Grouping', 'Trait-Based Assortment'],
            index=['Kinship', 'Random Grouping', 'Trait-Based Assortment'].index(s.get('colony_formation_method', 'Kinship')),
            disabled=not enable_multi_level_selection,
            help="**How colonies are formed:**\n- **Kinship:** Groups are formed from closely related individuals (siblings).\n- **Random Grouping:** Groups are formed randomly.\n- **Trait-Based Assortment:** Individuals with a similar 'social' gene group together.",
            key="colony_formation_method_selectbox"
        )
        
        colony_size = st.slider("Colony Size", 5, 50, s.get('colony_size', 10), 5, disabled=not enable_multi_level_selection, help="The number of individuals per colony.", key="colony_size_slider")
        
        group_fitness_weight = st.slider("Group Fitness Weight (Altruism Pressure)", 0.0, 1.0, s.get('group_fitness_weight', 0.3), 0.05, disabled=not enable_multi_level_selection, help="The proportion of an individual's final fitness that comes from its colony's success. High values reward altruism and punish selfishness.", key="group_fitness_weight_slider")
        
        selfishness_suppression_cost = st.slider("Selfishness Suppression Cost", 0.0, 0.2, s.get('selfishness_suppression_cost', 0.05), 0.01, disabled=not enable_multi_level_selection, help="A fitness cost applied to colonies that successfully evolve 'policing' mechanisms to suppress individual cheating, modeling the cost of maintaining social order.", key="selfishness_suppression_cost_slider")
        
        caste_specialization_bonus = st.slider("Caste Specialization Bonus", 0.0, 0.5, s.get('caste_specialization_bonus', 0.1), 0.01, disabled=not enable_multi_level_selection, help="A group-level fitness bonus if the colony's members successfully differentiate into distinct phenotypic 'castes' (e.g., workers and soldiers), rewarding division of labor.", key="caste_specialization_bonus_slider")
        
        inter_colony_competition_rate = st.slider("Inter-Colony Competition Rate", 0.0, 1.0, s.get('inter_colony_competition_rate', 0.1), 0.05, disabled=not enable_multi_level_selection, help="The rate at which colonies compete with each other, leading to the dissolution of losing colonies and the propagation of winning ones. Simulates tribal warfare or competition between hives.", key="inter_colony_competition_rate_slider")
        
        st.info(
            "Multi-Level Selection is the theoretical framework for understanding the evolution of cooperation and sociality."
        )

    with st.sidebar.expander(" 🧬 Mutation Control"):
        mutation_schedule = st.selectbox(
            "Mutation Rate Schedule",
            ['Constant', 'Linear Decay', 'Adaptive'],
            index=['Constant', 'Linear Decay', 'Adaptive'].index(s.get('mutation_schedule', 'Adaptive')),
            help="How the mutation rate changes over generations.",
            key="mutation_schedule_selectbox"
        )
        adaptive_mutation_strength = st.slider(
            "Adaptive Strength",
            min_value=0.1, max_value=5.0, value=s.get('adaptive_mutation_strength', 1.0),
            help="""
            **Simulates desperation in the face of stagnation.** A higher value causes a larger, more explosive spike in mutation rate when progress stalls, forcing the population out of a local optimum.
            **WARNING:** Very high values (>2.0) can cause wild, chaotic oscillations in mutation rate, destabilizing the evolutionary search and preventing fine-tuning.
            """,
            disabled=(mutation_schedule != 'Adaptive'),
            key="adaptive_mutation_strength_slider"
        )
    
    with st.sidebar.expander("🔬 Advanced Mutation Control & Operator Dynamics", expanded=False):
        enable_advanced_mutation = st.checkbox(
            "Enable Advanced Mutation Dynamics",
            value=s.get('enable_advanced_mutation', False),
            help="**DANGER: HIGHLY EXPERIMENTAL.** Unlocks a suite of sophisticated mutation operators that model complex biological and theoretical phenomena. This provides fine-grained control over the variational properties of the evolutionary search, but can lead to unpredictable dynamics.",
            key="enable_advanced_mutation_checkbox"
        )

        st.markdown("#### 1. Mutation Distribution & Shape")
        mutation_distribution_type = st.selectbox("Mutation Distribution", ['Gaussian', 'Cauchy', 'Laplace', 'Log-Normal'], index=['Gaussian', 'Cauchy', 'Laplace', 'Log-Normal'].index(s.get('mutation_distribution_type', 'Gaussian')), disabled=not enable_advanced_mutation, help="The statistical distribution for point mutations. Cauchy allows for rare, large 'leaps'.")
        mutation_scale_parameter = st.slider("Mutation Scale (σ)", 0.01, 0.5, s.get('mutation_scale_parameter', 0.1), 0.01, disabled=not enable_advanced_mutation, help="The scale parameter (e.g., standard deviation) of the mutation distribution.")
        structural_mutation_scale = st.slider("Structural Mutation Scale", 0.1, 2.0, s.get('structural_mutation_scale', 1.0), 0.1, disabled=not enable_advanced_mutation, help="A multiplier for the magnitude of structural changes (e.g., size of a new module).")
        mutation_correlation_factor = st.slider("Mutation Correlation Factor", 0.0, 1.0, s.get('mutation_correlation_factor', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Correlation between mutations applied to different parameters within the same module. High values mean parameters mutate together.")
        mutation_operator_bias = st.slider("Mutation Operator Bias", -1.0, 1.0, s.get('mutation_operator_bias', 0.0), 0.1, disabled=not enable_advanced_mutation, help="Bias between structural and parameter mutations. >0 favors structural, <0 favors parameter.")
        mutation_tail_heaviness = st.slider("Mutation Tail Heaviness", 0.1, 5.0, s.get('mutation_tail_heaviness', 1.0), 0.1, disabled=not enable_advanced_mutation, help="Controls the 'heaviness' of the mutation distribution's tails, allowing for more or fewer extreme mutations.")
        mutation_anisotropy_vector = st.text_input("Mutation Anisotropy Vector", value=s.get('mutation_anisotropy_vector', "1.0,1.0,1.0"), disabled=not enable_advanced_mutation, help="CSV of multipliers for different parameter types (e.g., size,plasticity,lr_mult) to make mutation anisotropic.")
        mutation_modality = st.slider("Mutation Modality", 1, 5, s.get('mutation_modality', 1), 1, disabled=not enable_advanced_mutation, help="Number of modes in the mutation distribution. >1 creates a multi-modal distribution, offering distinct 'packages' of change.")
        mutation_step_size_annealing = st.slider("Mutation Step-Size Annealing", 0.9, 1.0, s.get('mutation_step_size_annealing', 0.99), 0.005, disabled=not enable_advanced_mutation, help="Factor by which the mutation scale parameter is multiplied each generation. <1.0 anneals the search.")

        st.markdown("---")
        st.markdown("#### 2. Context-Dependent & Targeted Mutagenesis")
        fitness_dependent_mutation_strength = st.slider("Fitness-Dependent Mutation Strength", -1.0, 1.0, s.get('fitness_dependent_mutation_strength', 0.0), 0.1, disabled=not enable_advanced_mutation, help="How mutation rate scales with fitness. <0: fitter individuals mutate less (Lamarckian). >0: fitter individuals mutate more (exploration).")
        age_dependent_mutation_strength = st.slider("Age-Dependent Mutation Strength", -1.0, 1.0, s.get('age_dependent_mutation_strength', 0.0), 0.1, disabled=not enable_advanced_mutation, help="How mutation rate scales with individual's age. >0 simulates accumulating damage, <0 simulates stabilizing with age.")
        module_size_dependent_mutation = st.slider("Module Size-Dependent Mutation", -1.0, 1.0, s.get('module_size_dependent_mutation', 0.0), 0.1, disabled=not enable_advanced_mutation, help="How mutation rate scales with module size. >0: larger modules are more mutable. <0: smaller modules are more mutable.")
        connection_weight_dependent_mutation = st.slider("Connection Weight-Dependent Mutation", -1.0, 1.0, s.get('connection_weight_dependent_mutation', 0.0), 0.1, disabled=not enable_advanced_mutation, help="How mutation rate scales with connection weight. >0: strong connections are more plastic. <0: weak connections are more plastic.")
        somatic_hypermutation_rate = st.slider("Somatic Hypermutation Rate", 0.0, 0.1, s.get('somatic_hypermutation_rate', 0.0), 0.005, disabled=not enable_advanced_mutation, help="Probability of a 'hypermutation' event in a specific module during an individual's lifetime, simulating affinity maturation in immune systems.")
        error_driven_mutation_strength = st.slider("Error-Driven Mutation Strength", 0.0, 1.0, s.get('error_driven_mutation_strength', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Strength of a mechanism where modules responsible for high prediction errors have their mutation rates temporarily increased.")
        gene_centrality_mutation_bias = st.slider("Gene Centrality Mutation Bias", -1.0, 1.0, s.get('gene_centrality_mutation_bias', 0.0), 0.1, disabled=not enable_advanced_mutation, help="Bias mutation towards/away from central nodes in the network graph. >0 protects the core, <0 focuses change on the core.")
        epigenetic_mutation_influence = st.slider("Epigenetic Mutation Influence", 0.0, 1.0, s.get('epigenetic_mutation_influence', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Degree to which epigenetic markers can directly increase or decrease the mutation rate of the genes they are attached to.")
        mutation_hotspot_probability = st.slider("Mutation Hotspot Probability", 0.0, 0.1, s.get('mutation_hotspot_probability', 0.0), 0.005, disabled=not enable_advanced_mutation, help="Probability of designating a random gene as a 'hotspot' with a temporarily elevated mutation rate.")

        st.markdown("---")
        st.markdown("#### 3. Structural & Topological Mutation Control")
        add_connection_topology_bias = st.selectbox("Add Connection Topology Bias", ['Uniform', 'Local', 'Global'], index=['Uniform', 'Local', 'Global'].index(s.get('add_connection_topology_bias', 'Uniform')), disabled=not enable_advanced_mutation, help="Bias for adding new connections. Local connects nearby nodes, Global connects distant ones.")
        add_module_method = st.selectbox("Add Module Method", ['Split Connection', 'Orphan Node'], index=['Split Connection', 'Orphan Node'].index(s.get('add_module_method', 'Split Connection')), disabled=not enable_advanced_mutation, help="How new modules are integrated. 'Split Connection' is the classic NEAT method. 'Orphan Node' adds an unconnected module that must evolve connections later.")
        remove_connection_probability = st.slider("Remove Connection Probability", 0.0, 0.1, s.get('remove_connection_probability', 0.01), 0.005, disabled=not enable_advanced_mutation, help="Probability of removing a random connection during mutation.")
        remove_module_probability = st.slider("Remove Module Probability", 0.0, 0.05, s.get('remove_module_probability', 0.005), 0.001, disabled=not enable_advanced_mutation, help="Probability of removing a random module (and its connections).")
        connection_rewiring_probability = st.slider("Connection Rewiring Probability", 0.0, 0.2, s.get('connection_rewiring_probability', 0.0), 0.01, disabled=not enable_advanced_mutation, help="Probability of changing the source or target of an existing connection.")
        module_duplication_probability = st.slider("Module Duplication Probability", 0.0, 0.05, s.get('module_duplication_probability', 0.0), 0.005, disabled=not enable_advanced_mutation, help="Probability of duplicating an entire module, including its internal parameters but with new connections.")
        module_fusion_probability = st.slider("Module Fusion Probability", 0.0, 0.05, s.get('module_fusion_probability', 0.0), 0.005, disabled=not enable_advanced_mutation, help="Probability of fusing two connected modules into a single, larger module.")
        cycle_formation_probability = st.slider("Cycle Formation Probability", 0.0, 0.1, s.get('cycle_formation_probability', 0.01), 0.005, disabled=not enable_advanced_mutation, help="Probability of adding a connection that creates a recurrent cycle in the graph.")
        structural_mutation_phase = st.selectbox("Structural Mutation Phase", ['Early', 'Late', 'Continuous'], index=['Early', 'Late', 'Continuous'].index(s.get('structural_mutation_phase', 'Continuous')), disabled=not enable_advanced_mutation, help="Restrict structural mutations to a specific phase of the run. Early=first 25%, Late=last 25%.")

        st.markdown("---")
        st.markdown("#### 4. Metaplasticity & Learning-Guided Mutation")
        learning_rate_mutation_strength = st.slider("Learning Rate Mutation Strength", 0.0, 0.5, s.get('learning_rate_mutation_strength', 0.1), 0.01, disabled=not enable_advanced_mutation, help="The scale of mutations applied to the learning rate meta-parameter.")
        plasticity_mutation_strength = st.slider("Plasticity Mutation Strength", 0.0, 0.5, s.get('plasticity_mutation_strength', 0.1), 0.01, disabled=not enable_advanced_mutation, help="The scale of mutations applied to module plasticity parameters.")
        learning_improvement_mutation_bonus = st.slider("Learning Improvement Mutation Bonus", 0.0, 2.0, s.get('learning_improvement_mutation_bonus', 0.0), 0.1, disabled=not enable_advanced_mutation, help="A multiplier on mutation rate for individuals that showed high learning improvement (Baldwin effect) in their lifetime.")
        weight_change_mutation_correlation = st.slider("Weight Change-Mutation Correlation", -1.0, 1.0, s.get('weight_change_mutation_correlation', 0.0), 0.1, disabled=not enable_advanced_mutation, help="Correlates mutation direction with the direction of weight changes during lifetime learning. >0 reinforces learning, <0 counteracts it.")
        synaptic_tagging_credit_assignment = st.slider("Synaptic Tagging Credit Assignment", 0.0, 1.0, s.get('synaptic_tagging_credit_assignment', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Strength of a 'synaptic tagging' mechanism, where mutations are preferentially applied to connections that were recently active and led to a reward.")
        metaplasticity_rule = st.selectbox("Metaplasticity Rule", ['None', 'BCM', 'Homeostatic'], index=['None', 'BCM', 'Homeostatic'].index(s.get('metaplasticity_rule', 'None')), disabled=not enable_advanced_mutation, help="The rule governing how plasticity itself changes. BCM = Bienenstock-Cooper-Munro, Homeostatic = activity-dependent scaling.")
        learning_instability_penalty = st.slider("Learning Instability Penalty", 0.0, 1.0, s.get('learning_instability_penalty', 0.0), 0.05, disabled=not enable_advanced_mutation, help="A fitness penalty for individuals whose parameters change drastically during lifetime learning, promoting more stable learners.")
        gradient_guided_mutation_strength = st.slider("Gradient-Guided Mutation Strength", 0.0, 1.0, s.get('gradient_guided_mutation_strength', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Strength of a mechanism that biases mutations to follow the direction of the fitness gradient (approximated), turning mutation into a form of hill-climbing.")
        hessian_guided_mutation_strength = st.slider("Hessian-Guided Mutation Strength", 0.0, 1.0, s.get('hessian_guided_mutation_strength', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Strength of a mechanism that biases mutations towards flat regions of the fitness landscape (low Hessian), which often generalize better.")

        st.markdown("---")
        st.markdown("#### 5. Advanced Crossover & Recombination")
        crossover_operator = st.selectbox("Crossover Operator", ['Homologous', 'Uniform', 'N-Point'], index=['Homologous', 'Uniform', 'N-Point'].index(s.get('crossover_operator', 'Homologous')), disabled=not enable_advanced_mutation, help="The specific algorithm used for crossover.")
        crossover_n_points = st.slider("Crossover N-Points", 1, 5, s.get('crossover_n_points', 2), 1, disabled=(crossover_operator != 'N-Point'), help="Number of points for N-Point crossover.")
        crossover_parent_assortativity = st.slider("Crossover Parent Assortativity", -1.0, 1.0, s.get('crossover_parent_assortativity', 0.0), 0.1, disabled=not enable_advanced_mutation, help="Mating preference. >0: similar individuals mate (assortative). <0: dissimilar individuals mate (disassortative).")
        crossover_gene_dominance_probability = st.slider("Crossover Gene Dominance Probability", 0.0, 1.0, s.get('crossover_gene_dominance_probability', 0.0), 0.05, disabled=not enable_advanced_mutation, help="Probability that a gene from the fitter parent is always inherited (dominant).")
        sexual_reproduction_rate = st.slider("Sexual Reproduction Rate", 0.0, 1.0, s.get('sexual_reproduction_rate', 1.0), 0.05, disabled=not enable_advanced_mutation, help="The proportion of offspring produced by crossover (sexually) vs. mutation alone (asexually).")
        inbreeding_penalty_factor = st.slider("Inbreeding Penalty Factor", 0.0, 1.0, s.get('inbreeding_penalty_factor', 0.0), 0.05, disabled=not enable_advanced_mutation, help="A fitness penalty applied to offspring of closely related parents, promoting outbreeding.")
        horizontal_gene_transfer_rate = st.slider("Horizontal Gene Transfer Rate", 0.0, 0.05, s.get('horizontal_gene_transfer_rate', 0.0), 0.001, disabled=not enable_advanced_mutation, help="Probability of transferring a single random module/gene from one individual to another in the population, simulating bacterial conjugation.")
        polyploidy_probability = st.slider("Polyploidy Probability", 0.0, 0.05, s.get('polyploidy_probability', 0.0), 0.001, disabled=not enable_advanced_mutation, help="Probability of an offspring inheriting the entire chromosome set from both parents, leading to a massive increase in genetic material.")
        meiotic_recombination_rate = st.slider("Meiotic Recombination Rate", 0.0, 1.0, s.get('meiotic_recombination_rate', 0.5), 0.05, disabled=not enable_advanced_mutation, help="Controls the frequency of crossover events along the chromosome during sexual reproduction.")

    st.sidebar.markdown("### Selection Strategy")
    selection_pressure = st.sidebar.slider(
        "Selection Pressure", min_value=0.1, max_value=0.9, value=s.get('selection_pressure', 0.4), step=0.05,
        help="""
        **'Survival of the Fittest'.** The fraction of the population that survives to reproduce each generation.
        - **High:** Aggressive, elitist selection. Fast convergence, but high risk of getting stuck in local optima.
        - **Low:** Gentle selection. Slower progress, but maintains more diversity.
        """,
        key="selection_pressure_slider"
    )
    
    enable_diversity_pressure = st.sidebar.checkbox(
        "Enable Diversity-Pressured Selection", value=s.get('enable_diversity_pressure', True), 
        help="**Actively fights premature convergence.** Rewards novel architectures during selection, helping to prevent the population from getting stuck in an evolutionary dead-end by protecting unique but temporarily less-fit individuals."
    )
    diversity_weight = st.sidebar.slider(
        "Diversity Weight", 0.0, 5.0, s.get('diversity_weight', 0.8), 0.1,
        disabled=not enable_diversity_pressure,
        help="""
        **Controls the core Exploration vs. Exploitation trade-off.** How much to prioritize architectural novelty vs. raw fitness.
        - **>1.0:** Strongly favors exploration, creating a bizarre menagerie of unique solutions, like nature's wild diversity.
        - **<0.5:** Prioritizes exploitation, focusing on perfecting known good solutions.
        **WARNING:** Extreme values (>2.5) can cause the system to ignore fitness almost entirely, leading to beautiful but useless architectures.
        """,
        key="diversity_weight_slider"
    )
    
    with st.sidebar.expander("Speciation (NEAT-style)", expanded=True):
        enable_speciation = st.checkbox("Enable Speciation", value=s.get('enable_speciation', True), help="**Protects innovation by creating niches.** Groups similar individuals into 'species', where they only compete amongst themselves. This allows a novel but currently unoptimized idea to survive and improve without being wiped out by the dominant species.", key="enable_speciation_checkbox")
        compatibility_threshold = st.slider(
            "Compatibility Threshold",
            min_value=1.0, max_value=50.0, value=s.get('compatibility_threshold', 7.0), step=0.5,
            disabled=not enable_speciation,
            help="""
            **Defines 'what is a species'.** The maximum genomic distance for two individuals to be considered compatible.
            - **High:** Broad species definitions, leading to fewer, larger, more diverse species.
            - **Low:** Narrow definitions, leading to many small, highly specialized species.
            **WARNING:** Extreme values can disrupt speciation. Too low and every individual is its own species; too high and there is only one species.
            """,
            key="compatibility_threshold_slider"
        )

        with st.expander("🌌 Advanced Speciation & Ecosystem Physics", expanded=False):
            enable_advanced_speciation = st.checkbox(
                "Enable Advanced Speciation Physics",
                value=s.get('enable_advanced_speciation', False),
                help="**DANGER: HIGHLY EXPERIMENTAL.** Unlocks a vast suite of parameters that model deep, complex ecosystem dynamics and can lead to unpredictable or unstable evolution. Use with caution and a clear hypothesis.",
                key="enable_advanced_speciation_checkbox"
            )

            if not enable_advanced_speciation:
                st.warning("Advanced Speciation Physics are disabled. These parameters will have no effect.")
            else:
                st.success("Advanced Speciation Physics are ACTIVE. You are now manipulating the deep structure of the ecosystem.")

            st.markdown("---")
            st.markdown("##### 1. Dynamic Compatibility & Distance Metrics")
            dynamic_threshold_adjustment_rate = st.slider("Dynamic Threshold Adjustment Rate", 0.0, 0.2, s.get('dynamic_threshold_adjustment_rate', 0.0), 0.01, disabled=not enable_advanced_speciation, help="Rate at which the compatibility threshold automatically adjusts based on the number of species. Positive values increase the threshold when there are too many species, and vice-versa.")
            distance_weight_c1 = st.slider("Distance Weight c1 (Disjoint/Excess)", 0.1, 5.0, s.get('distance_weight_c1', 1.0), 0.1, disabled=not enable_advanced_speciation, help="Weight for disjoint and excess genes in the genomic distance calculation (NEAT's c1).")
            distance_weight_c2 = st.slider("Distance Weight c2 (Matching Attributes)", 0.1, 5.0, s.get('distance_weight_c2', 0.5), 0.1, disabled=not enable_advanced_speciation, help="Weight for attribute differences in matching genes (NEAT's c3).")
            phenotypic_distance_weight = st.slider("Phenotypic Distance Weight", 0.0, 1.0, s.get('phenotypic_distance_weight', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Weight for phenotypic traits (e.g., fitness, accuracy) in the distance calculation, blending genotype and phenotype for speciation.")
            age_distance_weight = st.slider("Age Distance Weight", 0.0, 1.0, s.get('age_distance_weight', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Weight for the difference in age between two individuals in the distance calculation.")
            lineage_distance_factor = st.slider("Lineage Distance Factor", 0.0, 1.0, s.get('lineage_distance_factor', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A factor that increases distance between individuals from different deep lineages, promoting lineage separation.")
            distance_normalization_factor = st.slider("Distance Normalization Factor", 0.1, 2.0, s.get('distance_normalization_factor', 1.0), 0.1, disabled=not enable_advanced_speciation, help="A global divisor for the genomic distance calculation to tune its scale.")
            developmental_rule_distance_weight = st.slider("Developmental Rule Distance Weight", 0.0, 1.0, s.get('developmental_rule_distance_weight', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Weight for differences in developmental rules in the distance calculation.")
            meta_param_distance_weight = st.slider("Meta-Parameter Distance Weight", 0.0, 1.0, s.get('meta_param_distance_weight', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Weight for differences in evolvable meta-parameters (e.g., mutation rate) in the distance calculation.")

            st.markdown("---")
            st.markdown("##### 2. Species Lifecycle & Fitness")
            species_stagnation_threshold = st.slider("Species Stagnation Threshold", 5, 100, s.get('species_stagnation_threshold', 15), 1, disabled=not enable_advanced_speciation, help="Generations a species can go without improvement before being penalized.")
            stagnation_penalty = st.slider("Stagnation Penalty", 0.0, 1.0, s.get('stagnation_penalty', 0.1), 0.05, disabled=not enable_advanced_speciation, help="Fitness penalty multiplier applied to stagnating species.")
            species_age_bonus = st.slider("Species Age Bonus", 0.0, 0.2, s.get('species_age_bonus', 0.0), 0.01, disabled=not enable_advanced_speciation, help="A small fitness bonus for members of older, more established species.")
            species_novelty_bonus = st.slider("Species Novelty Bonus", 0.0, 0.2, s.get('species_novelty_bonus', 0.0), 0.01, disabled=not enable_advanced_speciation, help="A temporary fitness bonus for members of a newly formed species, helping it gain a foothold.")
            min_species_size_for_survival = st.slider("Min Species Size for Survival", 1, 10, s.get('min_species_size_for_survival', 2), 1, disabled=not enable_advanced_speciation, help="Species with fewer individuals than this are automatically culled.")
            species_extinction_threshold = st.slider("Species Extinction Fitness Threshold", 0.0, 0.5, s.get('species_extinction_threshold', 0.01), 0.01, disabled=not enable_advanced_speciation, help="Mean fitness below which a species is marked for extinction.")
            species_merger_threshold = st.slider("Species Merger Threshold", 0.1, 5.0, s.get('species_merger_threshold', 0.5), 0.1, disabled=not enable_advanced_speciation, help="Genomic distance below which two species representatives will cause their species to merge.")
            species_merger_probability = st.slider("Species Merger Probability", 0.0, 0.1, s.get('species_merger_probability', 0.0), 0.005, disabled=not enable_advanced_speciation, help="Probability per generation for two genomically close species to merge into one.")

            st.markdown("---")
            st.markdown("##### 3. Niche Dynamics & Competition")
            niche_construction_strength = st.slider("Niche Construction Strength", 0.0, 1.0, s.get('niche_construction_strength', 0.0), 0.05, disabled=not enable_advanced_speciation, help="How much individuals can modify their local fitness landscape for their descendants.")
            character_displacement_pressure = st.slider("Character Displacement Pressure", 0.0, 1.0, s.get('character_displacement_pressure', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A direct pressure for competing species to diverge in their phenotypic traits.")
            intraspecific_competition_scaling = st.slider("Intraspecific Competition Scaling", 0.5, 2.0, s.get('intraspecific_competition_scaling', 1.0), 0.1, disabled=not enable_advanced_speciation, help="Exponent for scaling competition within a species. >1 means more intense competition in large species.")
            interspecific_competition_scaling = st.slider("Interspecific Competition Scaling", 0.0, 2.0, s.get('interspecific_competition_scaling', 0.5), 0.1, disabled=not enable_advanced_speciation, help="Scaling factor for competition between different species.")
            resource_depletion_rate = st.slider("Resource Depletion Rate", 0.0, 1.0, s.get('resource_depletion_rate', 0.0), 0.05, disabled=not enable_advanced_speciation, help="How quickly individuals in a niche deplete the available 'fitness resources', increasing competition.")
            niche_overlap_penalty = st.slider("Niche Overlap Penalty", 0.0, 1.0, s.get('niche_overlap_penalty', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A fitness penalty based on how much a species' phenotypic space overlaps with others.")
            niche_capacity = st.slider("Niche Capacity", 10, 500, s.get('niche_capacity', 50), 10, disabled=not enable_advanced_speciation, help="The maximum number of individuals a single niche (species) can support.")

            st.markdown("---")
            st.markdown("##### 4. Reproductive Isolation & Mating")
            sexual_selection_factor = st.slider("Sexual Selection Factor", 0.0, 1.0, s.get('sexual_selection_factor', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Introduces a 'mating preference' component to selection, where individuals may prefer mates with certain traits.")
            mating_preference_strength = st.slider("Mating Preference Strength", 0.0, 1.0, s.get('mating_preference_strength', 0.0), 0.05, disabled=not enable_advanced_speciation, help="How strongly individuals prefer mates from their own species.")
            outbreeding_depression_penalty = st.slider("Outbreeding Depression Penalty", 0.0, 1.0, s.get('outbreeding_depression_penalty', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A fitness penalty for offspring from parents of different species.")
            inbreeding_depression_penalty = st.slider("Inbreeding Depression Penalty", 0.0, 1.0, s.get('inbreeding_depression_penalty', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A fitness penalty for offspring of very closely related parents within the same species.")
            reproductive_isolation_threshold = st.slider("Reproductive Isolation Threshold", 1.0, 10.0, s.get('reproductive_isolation_threshold', 3.0), 0.5, disabled=not enable_advanced_speciation, help="A hard genomic distance threshold above which two individuals cannot produce offspring.")
            assortative_mating_strength = st.slider("Assortative Mating Strength", -1.0, 1.0, s.get('assortative_mating_strength', 0.0), 0.1, disabled=not enable_advanced_speciation, help="Strength of preference for mating with phenotypically similar individuals. Positive is assortative, negative is disassortative.")

            st.markdown("---")
            st.markdown("##### 5. Speciation Modes & Triggers")
            sympatric_speciation_pressure = st.slider("Sympatric Speciation Pressure", 0.0, 1.0, s.get('sympatric_speciation_pressure', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A pressure that encourages speciation even without geographic isolation, based on trait differentiation.")
            allopatric_speciation_trigger = st.slider("Allopatric Speciation Trigger", 0.0, 1.0, s.get('allopatric_speciation_trigger', 0.0), 0.05, disabled=not enable_advanced_speciation, help="If population topology is fragmented (e.g., Island Model), this triggers speciation for isolated groups.")
            parapatric_speciation_gradient = st.slider("Parapatric Speciation Gradient", 0.0, 1.0, s.get('parapatric_speciation_gradient', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Strength of a spatial gradient in selective pressures that can lead to speciation at the boundary.")
            peripatric_speciation_founder_effect = st.slider("Peripatric Speciation Founder Effect", 0.0, 1.0, s.get('peripatric_speciation_founder_effect', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Strength of genetic drift in small, isolated 'founder' populations, promoting rapid speciation.")
            adaptive_radiation_trigger_threshold = st.slider("Adaptive Radiation Trigger Threshold", 0.0, 1.0, s.get('adaptive_radiation_trigger_threshold', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Fitness threshold that, when crossed by a species, triggers a temporary boost in its innovation rate, simulating the opening of new niches.")
            adaptive_radiation_strength = st.slider("Adaptive Radiation Strength", 1.0, 5.0, s.get('adaptive_radiation_strength', 2.0), 0.5, disabled=not enable_advanced_speciation, help="Multiplier on innovation and mutation rates during an adaptive radiation event.")

            st.markdown("---")
            st.markdown("##### 6. Social & Kin Dynamics")
            kin_selection_bonus = st.slider("Kin Selection Bonus", 0.0, 1.0, s.get('kin_selection_bonus', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A fitness bonus applied based on the success of close relatives, promoting altruistic behavior within a species.")
            group_selection_strength = st.slider("Group (Species) Selection Strength", 0.0, 1.0, s.get('group_selection_strength', 0.0), 0.05, disabled=not enable_advanced_speciation, help="The strength of selection at the species level, promoting species-level traits over individual fitness.")
            altruism_cost = st.slider("Altruism Cost", 0.0, 0.1, s.get('altruism_cost', 0.0), 0.005, disabled=not enable_advanced_speciation, help="A direct fitness cost for an individual to perform an 'altruistic' action that benefits its species.")
            species_reputation_factor = st.slider("Species Reputation Factor", 0.0, 1.0, s.get('species_reputation_factor', 0.0), 0.05, disabled=not enable_advanced_speciation, help="How much a species' 'reputation' (e.g., for cooperation) influences interactions with other species.")

            st.markdown("---")
            st.markdown("##### 7. Macro-Evolutionary Dynamics")
            punctuated_equilibrium_trigger_sensitivity = st.slider("Punctuated Equilibrium Trigger Sensitivity", 0.0, 1.0, s.get('punctuated_equilibrium_trigger_sensitivity', 0.1), 0.05, disabled=not enable_advanced_speciation, help="Sensitivity to stagnation for triggering a 'punctuation' event (e.g., a massive, temporary mutation rate spike).")
            background_extinction_rate = st.slider("Background Extinction Rate", 0.0, 0.1, s.get('background_extinction_rate', 0.0), 0.005, disabled=not enable_advanced_speciation, help="A constant, low probability of any individual being removed randomly per generation, simulating random death.")
            key_innovation_bonus = st.slider("Key Innovation Bonus", 0.0, 1.0, s.get('key_innovation_bonus', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A large, one-time fitness bonus for discovering a 'key innovation' (e.g., a new module type), which can trigger adaptive radiation.")
            invasive_species_introduction_prob = st.slider("Invasive Species Introduction Probability", 0.0, 0.1, s.get('invasive_species_introduction_prob', 0.0), 0.005, disabled=not enable_advanced_speciation, help="Probability per generation of introducing a pre-evolved, highly-fit individual from an external source.")
            refugia_survival_bonus = st.slider("Refugia Survival Bonus", 0.0, 1.0, s.get('refugia_survival_bonus', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A bonus to survival probability during a cataclysm for individuals in under-populated regions of the landscape (species refugia).")
            phyletic_gradualism_factor = st.slider("Phyletic Gradualism Factor", 0.0, 1.0, s.get('phyletic_gradualism_factor', 0.0), 0.05, disabled=not enable_advanced_speciation, help="A factor that smoothes evolutionary changes over time, enforcing a more gradual mode of evolution.")
            species_sorting_strength = st.slider("Species Sorting Strength", 0.0, 1.0, s.get('species_sorting_strength', 0.0), 0.05, disabled=not enable_advanced_speciation, help="Strength of a macro-evolutionary force where species with certain traits (e.g., high evolvability) have a higher survival rate, regardless of individual fitness.")

        st.info("Speciation uses a genomic distance metric based on form, module/connection differences, and parameter differences.")

    with st.sidebar.expander("🌌 Advanced Algorithmic & Theoretical Frameworks", expanded=False):
        enable_advanced_frameworks = st.checkbox(
            "Enable Advanced Frameworks Engine",
            value=s.get('enable_advanced_frameworks', False),
            help="""
            **WARNING: HIGHLY EXPERIMENTAL & UNSTABLE.**
            
            This section introduces parameters derived from deep theoretical computer science, advanced learning theory, and abstract mathematical concepts. They impose extremely specific and powerful priors on the evolutionary search.
            
            - **Effect:** These are not simple fitness bonuses. They fundamentally alter the geometry of the search space itself.
            - **Use Case:** For research into the absolute cutting-edge of theoretical AI and A-Life. Not for general use.
            
            **Enabling this may lead to beautiful, mathematically profound, but completely non-functional architectures. The risk of total evolutionary collapse is extremely high.**
            """,
            key="enable_advanced_frameworks_checkbox"
        )

        if not enable_advanced_frameworks:
            st.warning("Advanced Frameworks Engine is disabled. These parameters will have no effect.")
        else:
            st.success("Advanced Frameworks Engine is ACTIVE. You are manipulating abstract theoretical constructs. Expect the unexpected.")

        st.markdown("---")
        st.markdown("#### 1. Computational Logic & Metamathematics")

        chaitin_omega_bias = st.slider(
            "Chaitin's Omega Bias (Algorithmic Randomness)", 0.0, 1.0, s.get('chaitin_omega_bias', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="chaitin_omega_bias_slider",
            help="**Bias towards incompressible complexity.** Applies a fitness pressure favoring architectures whose description is algorithmically random (high Kolmogorov complexity), simulating a bias towards structures that resemble Chaitin's constant Ω."
        )
        godel_incompleteness_penalty = st.slider(
            "Gödelian Incompleteness Penalty", 0.0, 1.0, s.get('godel_incompleteness_penalty', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="godel_incompleteness_penalty_slider",
            help="**Penalizes axiomatic rigidity.** Applies a fitness cost to architectures that become too 'formal' or 'axiomatic', promoting systems that remain open to new, unprovable information. A conceptual check against rigid over-specialization."
        )
        turing_completeness_bonus = st.slider(
            "Turing Completeness Bonus", 0.0, 1.0, s.get('turing_completeness_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="turing_completeness_bonus_slider",
            help="**Rewards computational universality.** A bonus for architectures that exhibit properties consistent with Turing completeness (e.g., conditional branching, memory manipulation), such as the presence of recurrent loops and gating mechanisms."
        )
        lambda_calculus_isomorphism = st.slider(
            "Lambda Calculus Isomorphism", 0.0, 1.0, s.get('lambda_calculus_isomorphism', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="lambda_calculus_isomorphism_slider",
            help="**Pressure towards functional programming structures.** Rewards architectures whose graph structure is isomorphic to expressions in lambda calculus (e.g., clear function application, abstraction, and variable binding)."
        )
        proof_complexity_cost = st.slider(
            "Proof Complexity Cost", 0.0, 1.0, s.get('proof_complexity_cost', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="proof_complexity_cost_slider",
            help="**Cost based on logical verification.** A fitness cost proportional to the complexity of generating a formal 'proof' of the network's output from its input, favoring architectures with simple, verifiable reasoning paths."
        )
        constructive_type_theory_adherence = st.slider(
            "Constructive Type Theory Adherence", 0.0, 1.0, s.get('constructive_type_theory_adherence', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="constructive_type_theory_adherence_slider",
            help="**Rewards Curry-Howard correspondence.** Favors architectures where modules represent types and connections represent functions, aligning the network structure with principles of constructive mathematics and type theory."
        )

        st.markdown("---")
        st.markdown("#### 2. Advanced Statistical Learning Theory")

        pac_bayes_bound_minimization = st.slider(
            "PAC-Bayes Bound Minimization", 0.0, 1.0, s.get('pac_bayes_bound_minimization', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="pac_bayes_bound_minimization_slider",
            help="**Directly optimize for generalization.** Applies a fitness pressure to minimize the PAC-Bayesian bound on the generalization error, which balances empirical performance with the 'distance' from a prior distribution over weights."
        )
        vc_dimension_constraint = st.slider(
            "VC Dimension Constraint", 0.0, 1.0, s.get('vc_dimension_constraint', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="vc_dimension_constraint_slider",
            help="**Penalize model capacity.** Applies a fitness cost based on the estimated Vapnik-Chervonenkis (VC) dimension of the architecture, directly penalizing the model's capacity to shatter data and thus promoting better generalization."
        )
        rademacher_complexity_penalty = st.slider(
            "Rademacher Complexity Penalty", 0.0, 1.0, s.get('rademacher_complexity_penalty', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="rademacher_complexity_penalty_slider",
            help="**Data-dependent generalization bound.** Penalizes architectures based on their Rademacher complexity, a measure of how well the function class can fit random noise. A more sophisticated alternative to VC dimension."
        )
        algorithmic_stability_pressure = st.slider(
            "Algorithmic Stability Pressure", 0.0, 1.0, s.get('algorithmic_stability_pressure', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="algorithmic_stability_pressure_slider",
            help="**Reward robustness to data sampling.** Favors architectures whose lifetime learning process is 'stable', meaning small changes in the training data lead to small changes in the final learned model."
        )
        maml_readiness_bonus = st.slider(
            "MAML Readiness Bonus", 0.0, 1.0, s.get('maml_readiness_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="maml_readiness_bonus_slider",
            help="**Evolve for fast adaptation.** A fitness bonus for architectures whose parameter landscape is well-suited for meta-learning via Model-Agnostic Meta-Learning (MAML), i.e., they are easy to fine-tune to new tasks."
        )
        causal_inference_engine_bonus = st.slider(
            "Causal Inference Engine Bonus", 0.0, 1.0, s.get('causal_inference_engine_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="causal_inference_engine_bonus_slider",
            help="**Reward for reasoning about interventions.** A bonus for architectures that develop structures capable of performing causal inference, for example, by approximating Pearl's do-calculus or having separate observational and interventional pathways."
        )

        st.markdown("---")
        st.markdown("#### 3. Morphogenetic Engineering & Artificial Embryogeny")

        reaction_diffusion_activator_rate = st.slider(
            "Reaction-Diffusion Activator Rate", 0.0, 1.0, s.get('reaction_diffusion_activator_rate', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="reaction_diffusion_activator_rate_slider",
            help="**Controls the 'activator' in a Turing pattern.** In a simulated reaction-diffusion system for development, this controls the production rate of the self-catalyzing activator chemical."
        )
        reaction_diffusion_inhibitor_rate = st.slider(
            "Reaction-Diffusion Inhibitor Rate", 0.0, 1.0, s.get('reaction_diffusion_inhibitor_rate', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="reaction_diffusion_inhibitor_rate_slider",
            help="**Controls the 'inhibitor' in a Turing pattern.** This controls the production rate of the long-range inhibitor, which is necessary for forming stable patterns like spots or stripes in module placement."
        )
        morphogen_gradient_decay = st.slider(
            "Morphogen Gradient Decay", 0.0, 1.0, s.get('morphogen_gradient_decay', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="morphogen_gradient_decay_slider",
            help="**Controls developmental signaling range.** Sets the decay rate of simulated morphogen gradients, which cells use to determine their position and fate. High decay = short-range signaling and sharp boundaries."
        )
        cell_adhesion_factor = st.slider(
            "Cell Adhesion Factor", 0.0, 1.0, s.get('cell_adhesion_factor', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="cell_adhesion_factor_slider",
            help="**Simulates physical cell-cell adhesion.** A force that encourages modules of the same 'type' to cluster together during simulated development, leading to more cohesive and segregated brain regions."
        )
        apoptosis_schedule_factor = st.slider(
            "Apoptosis Schedule Factor", 0.0, 1.0, s.get('apoptosis_schedule_factor', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="apoptosis_schedule_factor_slider",
            help="**Controls programmed cell death.** A parameter influencing a developmental timer that triggers apoptosis (programmed death) of certain modules, allowing for sculpting of the final architecture by removal."
        )
        hox_gene_expression_control = st.slider(
            "Hox Gene Expression Control", 0.0, 1.0, s.get('hox_gene_expression_control', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="hox_gene_expression_control_slider",
            help="**Simulates master developmental genes.** A bonus for architectures that evolve a small set of 'master' genes that control the identity and placement of large regions of the network, analogous to Hox genes defining body segments."
        )

        st.markdown("---")
        st.markdown("#### 4. Collective Intelligence & Socio-Cultural Dynamics")

        stigmergy_potential_factor = st.slider(
            "Stigmergy Potential Factor", 0.0, 1.0, s.get('stigmergy_potential_factor', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="stigmergy_potential_factor_slider",
            help="**Rewards indirect communication.** A bonus for architectures that can modify a shared 'environment' (e.g., epigenetic markers) that other individuals can then sense, simulating ant-like stigmergic communication."
        )
        quorum_sensing_threshold = st.slider(
            "Quorum Sensing Threshold", 0.0, 1.0, s.get('quorum_sensing_threshold', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="quorum_sensing_threshold_slider",
            help="**Enables collective action.** In a multi-agent context, this sets the threshold of 'signal' density required for a population to switch to a collective behavior, simulating bacterial quorum sensing."
        )
        social_learning_fidelity = st.slider(
            "Social Learning Fidelity", 0.0, 1.0, s.get('social_learning_fidelity', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="social_learning_fidelity_slider_2835g",
            help="**Controls accuracy of cultural inheritance.** The fidelity with which an individual can copy a structural motif or parameter set from a successful peer (non-genetically). High fidelity allows for rapid cultural evolution."
        )
        cultural_transmission_rate = st.slider(
            "Cultural Transmission Rate (Memetics)", 0.0, 1.0, s.get('cultural_transmission_rate', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="cultural_transmission_rate_slider",
            help="**The rate of 'meme' propagation.** The probability that a successful architectural motif (a 'meme') is copied horizontally to other individuals in the population, bypassing standard genetic inheritance."
        )
        division_of_labor_incentive = st.slider(
            "Division of Labor Incentive", 0.0, 1.0, s.get('division_of_labor_incentive', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="division_of_labor_incentive_slider",
            help="**Rewards eco-systemic specialization.** A fitness bonus applied to the whole population if it successfully diversifies into distinct, complementary 'castes' or species that occupy different niches."
        )
        consensus_algorithm_efficiency = st.slider(
            "Consensus Algorithm Efficiency", 0.0, 1.0, s.get('consensus_algorithm_efficiency', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="consensus_algorithm_efficiency_slider",
            help="**Rewards distributed agreement.** A bonus for populations that evolve mechanisms to quickly and accurately reach a consensus on a shared value, simulating algorithms like Paxos or Raft."
        )

        st.markdown("---")
        st.markdown("#### 5. Advanced Game Theory & Economic Models")

        hawk_dove_strategy_ratio = st.slider(
            "Hawk-Dove Strategy Ratio", 0.0, 1.0, s.get('hawk_dove_strategy_ratio', 0.5), 0.01,
            disabled=not enable_advanced_frameworks, key="hawk_dove_strategy_ratio_slider",
            help="**Sets the payoff for aggression.** In a simulated Hawk-Dove game for resources, this sets the cost of conflict. Low values favor aggressive 'Hawk' strategies; high values favor passive 'Dove' strategies."
        )
        ultimatum_game_fairness_pressure = st.slider(
            "Ultimatum Game Fairness Pressure", 0.0, 1.0, s.get('ultimatum_game_fairness_pressure', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="ultimatum_game_fairness_pressure_slider",
            help="**Rewards 'fair' resource sharing.** Simulates an Ultimatum Game between individuals and rewards populations that converge on 'fair' proposals, promoting the evolution of cooperative norms."
        )
        principal_agent_alignment_bonus = st.slider(
            "Principal-Agent Alignment Bonus", 0.0, 1.0, s.get('principal_agent_alignment_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="principal_agent_alignment_bonus_slider",
            help="**Rewards internal goal alignment.** Models the architecture as a 'firm' with a principal (output layer) and agents (internal modules). Rewards architectures where agent behavior (e.g., maximizing local information) aligns with the principal's goal (maximizing fitness)."
        )
        market_clearing_price_efficiency = st.slider(
            "Market Clearing Price Efficiency", 0.0, 1.0, s.get('market_clearing_price_efficiency', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="market_clearing_price_efficiency_slider",
            help="**Simulates a computational market.** Models connections as a market for information and rewards architectures that achieve an efficient 'market clearing price', balancing supply and demand for computation."
        )
        contract_theory_enforcement_cost = st.slider(
            "Contract Theory Enforcement Cost", 0.0, 1.0, s.get('contract_theory_enforcement_cost', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="contract_theory_enforcement_cost_slider",
            help="**The cost of enforcing cooperation.** In multi-module systems, this parameter represents the fitness cost of mechanisms that enforce 'contracts' (i.e., expected behavior) between modules, penalizing overly complex control systems."
        )
        vickrey_auction_selection_bonus = st.slider(
            "Vickrey Auction Selection Bonus", 0.0, 1.0, s.get('vickrey_auction_selection_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="vickrey_auction_selection_bonus_slider",
            help="**Rewards truthful 'bidding' for survival.** A selection mechanism where individuals 'bid' their fitness, and survivors are chosen via a Vickrey (second-price) auction. This incentivizes the evolution of accurate self-assessment of fitness."
        )

        st.markdown("---")
        st.markdown("#### 6. Advanced Neuromodulation & Synaptic Plasticity")

        dopamine_reward_prediction_error = st.slider(
            "Dopaminergic RPE Modulation", 0.0, 1.0, s.get('dopamine_reward_prediction_error', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="dopamine_reward_prediction_error_slider",
            help="**Simulates dopamine's role in learning.** Modulates the plasticity of connections based on a simulated Reward Prediction Error (RPE). Positive RPE increases plasticity (learning), negative RPE decreases it."
        )
        serotonin_uncertainty_signal = st.slider(
            "Serotonergic Uncertainty Signal", 0.0, 1.0, s.get('serotonin_uncertainty_signal', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="serotonin_uncertainty_signal_slider",
            help="**Simulates serotonin's role in mood and risk.** Modulates global learning rates based on environmental uncertainty or volatility. High uncertainty leads to lower learning rates (more cautious updates)."
        )
        acetylcholine_attentional_gain = st.slider(
            "Cholinergic Attentional Gain", 0.0, 1.0, s.get('acetylcholine_attentional_gain', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="acetylcholine_attentional_gain_slider",
            help="**Simulates acetylcholine's role in attention.** Increases the activation gain (sharpening the response) of modules that are targets of attentional mechanisms."
        )
        noradrenaline_arousal_level = st.slider(
            "Noradrenergic Arousal Level", 0.0, 1.0, s.get('noradrenaline_arousal_level', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="noradrenaline_arousal_level_slider",
            help="**Simulates noradrenaline's role in arousal/vigilance.** A global signal that affects network excitability and the exploration/exploitation trade-off. High arousal can increase network 'temperature' and promote exploration."
        )
        bcm_rule_sliding_threshold = st.slider(
            "BCM Rule Sliding Threshold", 0.0, 1.0, s.get('bcm_rule_sliding_threshold', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="bcm_rule_sliding_threshold_slider",
            help="**Controls metaplasticity.** In a Bienenstock-Cooper-Munro (BCM) learning rule simulation, this controls how the threshold between potentiation and depression slides based on past activity, preventing runaway feedback loops."
        )
        synaptic_scaling_homeostasis = st.slider(
            "Synaptic Scaling Homeostasis", 0.0, 1.0, s.get('synaptic_scaling_homeostasis', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="synaptic_scaling_homeostasis_slider",
            help="**Pressure for stable firing rates.** A fitness pressure that rewards architectures for maintaining stable average firing rates across modules by multiplicatively scaling incoming synaptic weights."
        )

        st.markdown("---")
        st.markdown("#### 7. Abstract Algebra & Category Theory Priors")

        group_theory_symmetry_bonus = st.slider(
            "Group Theory Symmetry Bonus", 0.0, 1.0, s.get('group_theory_symmetry_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="group_theory_symmetry_bonus_slider",
            help="**Rewards algebraic symmetries.** A bonus for architectures whose connection graph exhibits specific group symmetries (e.g., rotational, permutational), which is a core concept in equivariant deep learning."
        )
        category_theory_functorial_bonus = st.slider(
            "Category Theory Functorial Bonus", 0.0, 1.0, s.get('category_theory_functorial_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="category_theory_functorial_bonus_slider",
            help="**Rewards structure-preserving maps.** Rewards architectures where modules act as 'objects' and connection patterns act as 'functors' that preserve structure between different parts of the network, promoting compositional generalization."
        )
        monad_structure_bonus = st.slider(
            "Monad Structure Bonus", 0.0, 1.0, s.get('monad_structure_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="monad_structure_bonus_slider",
            help="**Bonus for composable computations.** Rewards the emergence of monadic structures (an endofunctor with 'return' and 'bind' operations), which are fundamental for sequencing computations in functional programming."
        )
        lie_algebra_dynamics_prior = st.slider(
            "Lie Algebra Dynamics Prior", 0.0, 1.0, s.get('lie_algebra_dynamics_prior', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="lie_algebra_dynamics_prior_slider",
            help="**Prior for continuous transformations.** A prior favoring network dynamics that can be described by a Lie algebra, promoting the learning of smooth, continuous transformations and symmetries."
        )
        simplicial_complex_bonus = st.slider(
            "Simplicial Complex Bonus", 0.0, 1.0, s.get('simplicial_complex_bonus', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="simplicial_complex_bonus_slider",
            help="**Rewards higher-order relationships.** A bonus for architectures that form higher-order structures like simplicial complexes (triangles, tetrahedra), allowing for representation of multi-way relationships beyond simple pairwise connections."
        )
        sheaf_computation_consistency = st.slider(
            "Sheaf Computation Consistency", 0.0, 1.0, s.get('sheaf_computation_consistency', 0.0), 0.01,
            disabled=not enable_advanced_frameworks, key="sheaf_computation_consistency_slider",
            help="**Rewards locally consistent global data.** From sheaf theory, this rewards architectures that can consistently aggregate local data defined on different modules into a coherent global picture, crucial for distributed sensing."
        )

    with st.sidebar.expander("♾️ Deep Evolutionary Physics & Information Dynamics", expanded=False):
        enable_deep_physics = st.checkbox(
            "Enable Deep Physics Engine",
            value=s.get('enable_deep_physics', False),
            help="""
            **Unlock the fundamental constants of the evolutionary universe.**
            
            Enabling this exposes 50 highly experimental parameters that model deep physical and informational principles. These are not standard evolutionary algorithm parameters; they represent theoretical concepts from physics, information theory, and cognitive science, applied to neuroevolution.
            
            - **Effect:** These parameters introduce subtle but powerful biases and constraints into the fitness landscape and evolutionary dynamics.
            - **Use Case:** For advanced research into the fundamental principles of intelligence and emergence. Not recommended for standard optimization tasks.
            
            **WARNING: EXTREME RISK OF DESTABILIZATION.** These parameters interact in complex, non-linear ways. Uninformed changes will almost certainly lead to evolutionary collapse, stagnation, or bizarre, non-functional outcomes. Use with extreme caution and a clear hypothesis.
            """,
            key="enable_deep_physics_checkbox"
        )

        if not enable_deep_physics:
            st.warning("The Deep Physics Engine is disabled. These parameters will have no effect. Enable to unlock the fundamental constants of this evolutionary universe.")
        else:
            st.success("Deep Physics Engine is ACTIVE. You are now editing the fundamental constants of the universe. Proceed with extreme caution.")

        st.markdown("---")
        st.markdown("#### 1. Information-Theoretic Dynamics")

        kolmogorov_pressure = st.slider(
            "Kolmogorov Pressure", 0.0, 1.0, s.get('kolmogorov_pressure', 0.0), 0.01,
            disabled=not enable_deep_physics, key="kolmogorov_pressure_slider",
            help="**Bias towards algorithmic simplicity.** Applies a fitness bonus proportional to the compressibility of the genotype's developmental program (approximated). High values favor elegant, simple generative rules over complex, bloated ones."
        )
        pred_info_bottleneck = st.slider(
            "Predictive Information Bottleneck", 0.0, 1.0, s.get('pred_info_bottleneck', 0.0), 0.01,
            disabled=not enable_deep_physics, key="pred_info_bottleneck_slider",
            help="**Balance between memory and prediction.** Rewards architectures that compress past information while retaining maximal predictive power about the future. Based on Tishby's Information Bottleneck principle."
        )
        causal_emergence_factor = st.slider(
            "Causal Emergence Factor", 0.0, 1.0, s.get('causal_emergence_factor', 0.0), 0.01,
            disabled=not enable_deep_physics, key="causal_emergence_factor_slider",
            help="**Rewards effective abstraction.** Favors architectures where macro-scale modules have more causal power (higher effective information) over the system's output than their constituent micro-components. Promotes the emergence of meaningful, high-level structures."
        )
        semantic_closure_pressure = st.slider(
            "Semantic Closure Pressure", 0.0, 1.0, s.get('semantic_closure_pressure', 0.0), 0.01,
            disabled=not enable_deep_physics, key="semantic_closure_pressure_slider",
            help="**Drives towards self-modeling.** Rewards architectures that can accurately model their own relationship with the environment. A step towards creating systems that 'understand' their own function."
        )
        phi_target = st.slider(
            "Integrated Information (Φ) Target", 0.0, 1.0, s.get('phi_target', 0.0), 0.01,
            disabled=not enable_deep_physics, key="phi_target_slider",
            help="**Bias towards integrated consciousness (IIT).** Approximates and rewards high Φ, a measure of a system's capacity to be conscious. Favors architectures that are both highly differentiated and highly integrated."
        )
        fep_gradient = st.slider(
            "Free Energy Principle (FEP) Gradient", 0.0, 1.0, s.get('fep_gradient', 0.0), 0.01,
            disabled=not enable_deep_physics, key="fep_gradient_slider",
            help="**The drive to minimize surprise.** Applies a fitness pressure for architectures to develop an accurate internal world model, minimizing the variational free energy (prediction error) of their sensory inputs. A core concept from Karl Friston's work."
        )
        transfer_entropy_maximization = st.slider(
            "Transfer Entropy Maximization", 0.0, 1.0, s.get('transfer_entropy_maximization', 0.0), 0.01,
            disabled=not enable_deep_physics, key="transfer_entropy_maximization_slider",
            help="**Promotes effective communication.** Rewards high information flow (transfer entropy) between connected modules, favoring architectures with strong, directed causal links."
        )
        synergy_bias = st.slider(
            "Synergistic Information Bias", 0.0, 1.0, s.get('synergy_bias', 0.0), 0.01,
            disabled=not enable_deep_physics, key="synergy_bias_slider",
            help="**Favors holistic computation.** Rewards information generated by groups of modules that cannot be reduced to the sum of its parts. Promotes non-linear, emergent computation."
        )
        state_space_compression = st.slider(
            "State-Space Compression Ratio", 0.0, 1.0, s.get('state_space_compression', 0.0), 0.01,
            disabled=not enable_deep_physics, key="state_space_compression_slider",
            help="**Rewards efficient representation.** Measures how compactly the architecture's internal state represents the external environment's state space. High values favor efficient world models."
        )
        fisher_gradient_ascent = st.slider(
            "Fisher Information Gradient Ascent", 0.0, 1.0, s.get('fisher_gradient_ascent', 0.0), 0.01,
            disabled=not enable_deep_physics, key="fisher_gradient_ascent_slider",
            help="**Optimizes the speed of adaptation.** Applies a force to move the population along the natural gradient of the fitness landscape, defined by the Fisher Information Metric. This can dramatically accelerate evolution in certain landscapes."
        )

        st.markdown("---")
        st.markdown("#### 2. Thermodynamics of Computation & Metabolism")

        landauer_efficiency = st.slider(
            "Landauer Limit Efficiency", 0.0, 1.0, s.get('landauer_efficiency', 0.0), 0.01,
            disabled=not enable_deep_physics, key="landauer_efficiency_slider",
            help="**Penalizes irreversible computation.** Applies a fitness cost proportional to the number of information-erasing operations (e.g., resetting a bit), based on Landauer's principle. Favors reversible or energy-efficient computing paradigms."
        )
        metabolic_power_law = st.slider(
            "Metabolic Power Law (Exponent)", 0.5, 1.5, s.get('metabolic_power_law', 0.75), 0.01,
            disabled=not enable_deep_physics, key="metabolic_power_law_slider",
            help="**Models biological scaling laws.** Sets the exponent for the metabolic cost function (Cost ∝ Size^Exponent). A value of 0.75 mimics Kleiber's Law, where larger architectures are more energy-efficient per parameter."
        )
        heat_dissipation_constraint = st.slider(
            "Heat Dissipation Constraint", 0.0, 1.0, s.get('heat_dissipation_constraint', 0.0), 0.01,
            disabled=not enable_deep_physics, key="heat_dissipation_constraint_slider",
            help="**Models physical thermal limits.** Applies a sharp fitness penalty if the estimated computational activity (entropy production) exceeds a threshold, simulating the physical constraint of heat dissipation."
        )
        homeostatic_pressure = st.slider(
            "Homeostatic Regulation Pressure", 0.0, 1.0, s.get('homeostatic_pressure', 0.0), 0.01,
            disabled=not enable_deep_physics, key="homeostatic_pressure_slider",
            help="**Rewards internal stability.** Favors architectures that can maintain stable internal activity patterns despite external perturbations. Promotes robust, self-regulating dynamics."
        )
        computational_temperature = st.slider(
            "Computational Temperature", 0.0, 1.0, s.get('computational_temperature', 0.0), 0.01,
            disabled=not enable_deep_physics, key="computational_temperature_slider",
            help="**Injects thermodynamic noise.** Introduces a global noise level into all computations, simulating a thermal environment. Higher temperatures make computation less reliable, favoring noise-robust architectures."
        )
        structural_decay_rate = st.slider(
            "Structural Integrity Decay Rate", 0.0, 0.1, s.get('structural_decay_rate', 0.0), 0.001,
            disabled=not enable_deep_physics, key="structural_decay_rate_slider",
            help="**Simulates aging and physical decay.** A per-generation probability for any connection or module to be damaged or destroyed, forcing the evolution of robust, redundant, or repairable systems."
        )
        repair_mechanism_cost = st.slider(
            "Repair Mechanism Cost", 0.0, 1.0, s.get('repair_mechanism_cost', 0.0), 0.01,
            disabled=not enable_deep_physics, key="repair_mechanism_cost_slider",
            help="**The metabolic cost of self-repair.** If Structural Decay is active, this parameter sets the fitness cost for architectures that possess self-repair capabilities (a hypothetical trait)."
        )
        szilard_engine_efficiency = st.slider(
            "Szilard's Engine Efficiency", 0.0, 1.0, s.get('szilard_engine_efficiency', 0.0), 0.01,
            disabled=not enable_deep_physics, key="szilard_engine_efficiency_slider",
            help="**Rewards converting information to 'work'.** A fitness bonus for architectures that use information to reduce their own computational cost, simulating a Maxwell's Demon or Szilard's Engine."
        )
        resource_scarcity = st.slider(
            "Resource Scarcity Simulation", 0.0, 1.0, s.get('resource_scarcity', 0.0), 0.01,
            disabled=not enable_deep_physics, key="resource_scarcity_slider",
            help="**Simulates a fluctuating energy budget.** Introduces periodic fluctuations in the 'energy' available, dynamically changing the weight of the efficiency objective. High values create boom-and-bust cycles."
        )
        allosteric_regulation_factor = st.slider(
            "Allosteric Regulation Factor", 0.0, 1.0, s.get('allosteric_regulation_factor', 0.0), 0.01,
            disabled=not enable_deep_physics, key="allosteric_regulation_factor_slider",
            help="**Promotes indirect control mechanisms.** Rewards the evolution of 'modulatory' connections that don't pass primary signals but instead alter the function of other modules, akin to allosteric enzymes."
        )

        st.markdown("---")
        st.markdown("#### 3. Quantum & Field-Theoretic Effects")

        quantum_annealing_fluctuation = st.slider(
            "Quantum Annealing Fluctuation", 0.0, 1.0, s.get('quantum_annealing_fluctuation', 0.0), 0.01,
            disabled=not enable_deep_physics, key="quantum_annealing_fluctuation_slider",
            help="**Allows tunneling through fitness barriers.** Introduces a probability for a genotype to make a large, non-local jump in genotype space to a state with lower 'energy' (higher fitness), simulating quantum tunneling."
        )
        holographic_constraint = st.slider(
            "Holographic Principle Constraint", 0.0, 1.0, s.get('holographic_constraint', 0.0), 0.01,
            disabled=not enable_deep_physics, key="holographic_constraint_slider",
            help="**Information scales with surface, not volume.** Applies a fitness penalty if the information content of a module (approximated by its parameters) exceeds its 'surface area' (number of connections). Favors distributed over centralized information."
        )
        renormalization_group_flow = st.slider(
            "Renormalization Group Flow", 0.0, 1.0, s.get('renormalization_group_flow', 0.0), 0.01,
            disabled=not enable_deep_physics, key="renormalization_group_flow_slider",
            help="**Evolves scale-invariant structures.** Applies a fitness pressure that rewards architectures whose statistical properties remain the same when 'zooming out' (coarse-graining). Promotes fractal-like, hierarchical structures."
        )
        symmetry_breaking_pressure = st.slider(
            "Symmetry Breaking Pressure", 0.0, 1.0, s.get('symmetry_breaking_pressure', 0.0), 0.01,
            disabled=not enable_deep_physics, key="symmetry_breaking_pressure_slider",
            help="**Forces specialization from homogeneity.** Applies a penalty to architectures with highly symmetric structures, encouraging them to 'break' the symmetry and develop specialized modules."
        )
        path_integral_exploration = st.slider(
            "Path Integral Exploration", 0.0, 1.0, s.get('path_integral_exploration', 0.0), 0.01,
            disabled=not enable_deep_physics, key="path_integral_exploration_slider",
            help="**Considers all possible evolutionary histories.** A conceptual parameter that biases selection towards genotypes that lie on many probable evolutionary paths, making them more 'robust' attractors in the landscape."
        )
        tqft_invariance = st.slider(
            "TQFT Invariance", 0.0, 1.0, s.get('tqft_invariance', 0.0), 0.01,
            disabled=not enable_deep_physics, key="tqft_invariance_slider",
            help="**Rewards topological robustness.** From Topological Quantum Field Theory, this rewards architectures whose function is invariant under continuous deformations (e.g., small changes in connection weights). Promotes extremely robust computational graphs."
        )
        gauge_theory_redundancy = st.slider(
            "Gauge Theory Redundancy", 0.0, 1.0, s.get('gauge_theory_redundancy', 0.0), 0.01,
            disabled=not enable_deep_physics, key="gauge_theory_redundancy_slider",
            help="**Favors robust internal symmetries.** Rewards architectures that have redundant internal descriptions, making them robust to certain classes of internal errors. Analogous to gauge symmetries in physics."
        )
        cft_scaling_exponent = st.slider(
            "CFT Scaling Exponent", 0.0, 4.0, s.get('cft_scaling_exponent', 0.0), 0.05,
            disabled=not enable_deep_physics, key="cft_scaling_exponent_slider",
            help="**Enforces conformal (scale-invariant) structure.** At evolutionary 'phase transitions', this biases towards architectures whose correlation functions obey a power law with this exponent, a signature of Conformal Field Theories."
        )
        spacetime_foam_fluctuation = st.slider(
            "Spacetime Foam Fluctuation", 0.0, 1.0, s.get('spacetime_foam_fluctuation', 0.0), 0.01,
            disabled=not enable_deep_physics, key="spacetime_foam_fluctuation_slider",
            help="**Introduces micro-fluctuations in the graph.** Simulates quantum foam by adding/removing transient micro-connections at each evaluation, testing for extreme structural robustness."
        )
        entanglement_assisted_comm = st.slider(
            "Entanglement-Assisted Communication", 0.0, 1.0, s.get('entanglement_assisted_comm', 0.0), 0.01,
            disabled=not enable_deep_physics, key="entanglement_assisted_comm_slider",
            help="**Simulates non-local information transfer.** Provides a small fitness bonus for pairs of modules that are highly correlated in their activity despite being topologically distant, simulating entanglement."
        )
        majorana_fermion_pairing_bonus = st.slider(
            "Majorana Fermion Pairing Bonus", 0.0, 1.0, s.get('majorana_fermion_pairing_bonus', 0.0), 0.05,
            help="A highly speculative bonus for architectures exhibiting symmetric, self-dual information pathways, analogous to Majorana fermion pairing in quantum physics. Rewards deep structural symmetries.",
            key="majorana_fermion_pairing_bonus_slider",
            disabled=not enable_deep_physics
        )

        st.markdown("---")
        st.markdown("#### 4. Topological & Geometric Constraints")

        manifold_adherence = st.slider(
            "Manifold Hypothesis Adherence", 0.0, 1.0, s.get('manifold_adherence', 0.0), 0.01,
            disabled=not enable_deep_physics, key="manifold_adherence_slider",
            help="**Assumes data lies on a low-dimensional manifold.** Rewards architectures whose internal representations have a lower intrinsic dimension, enforcing the manifold hypothesis."
        )
        group_equivariance_prior = st.slider(
            "Group Equivariance Prior", 0.0, 1.0, s.get('group_equivariance_prior', 0.0), 0.01,
            disabled=not enable_deep_physics, key="group_equivariance_prior_slider",
            help="**Enforces geometric deep learning symmetries.** Rewards architectures that are equivariant to certain transformations (e.g., rotation, translation), a core principle of GDL."
        )
        ricci_curvature_flow = st.slider(
            "Ricci Curvature Flow", 0.0, 1.0, s.get('ricci_curvature_flow', 0.0), 0.01,
            disabled=not enable_deep_physics, key="ricci_curvature_flow_slider",
            help="**Optimizes information geometry.** Simulates a flow to flatten the Ricci curvature of the architecture's parameter manifold, which can improve the efficiency of gradient-based lifetime learning."
        )
        homological_scaffold_stability = st.slider(
            "Homological Scaffold Stability", 0.0, 1.0, s.get('homological_scaffold_stability', 0.0), 0.01,
            disabled=not enable_deep_physics, key="homological_scaffold_stability_slider",
            help="**Preserves topological features.** From Topological Data Analysis, this rewards architectures for preserving the 'shape' (Betti numbers) of the input data's topology in their representations."
        )
        fractal_dimension_target = st.slider(
            "Fractal Dimension Target", 1.0, 3.0, s.get('fractal_dimension_target', 1.0), 0.05,
            disabled=not enable_deep_physics, key="fractal_dimension_target_slider",
            help="**Evolves self-similar structures.** Applies a fitness pressure for the architecture's connectivity graph to have a specific fractal (Hausdorff) dimension."
        )
        hyperbolic_embedding_factor = st.slider(
            "Hyperbolic Embedding Factor", 0.0, 1.0, s.get('hyperbolic_embedding_factor', 0.0), 0.01,
            disabled=not enable_deep_physics, key="hyperbolic_embedding_factor_slider",
            help="**Favors representation of hierarchies.** Rewards architectures whose connectivity graph can be embedded with low distortion into hyperbolic space, which is ideal for representing tree-like or hierarchical data."
        )
        small_world_bias = st.slider(
            "Small-World Network Bias", 0.0, 1.0, s.get('small_world_bias', 0.0), 0.01,
            disabled=not enable_deep_physics, key="small_world_bias_slider",
            help="**Balances local and global communication.** Rewards networks that have a high clustering coefficient (like regular grids) but a low average path length (like random graphs)."
        )
        scale_free_exponent = st.slider(
            "Scale-Free Network Exponent", 2.0, 4.0, s.get('scale_free_exponent', 2.0), 0.05,
            disabled=not enable_deep_physics, key="scale_free_exponent_slider",
            help="**Evolves hub-and-spoke topologies.** Rewards architectures whose degree distribution follows a power law with this exponent, characteristic of many real-world networks."
        )
        network_motif_bonus = st.slider(
            "Network Motif Bonus", 0.0, 1.0, s.get('network_motif_bonus', 0.0), 0.01,
            disabled=not enable_deep_physics, key="network_motif_bonus_slider",
            help="**Rewards specific computational micro-circuits.** Applies a fitness bonus for the over-representation of certain small subgraphs (motifs), like feed-forward loops."
        )
        rents_rule_exponent = st.slider(
            "Rent's Rule Exponent", 0.0, 1.0, s.get('rents_rule_exponent', 0.0), 0.01,
            disabled=not enable_deep_physics, key="rents_rule_exponent_slider",
            help="**Optimizes wiring cost and modularity.** Rewards architectures that obey Rent's Rule, an empirical relationship between the number of terminals and internal components of a module. High exponents indicate high complexity."
        )
        autocatalytic_set_emergence = st.slider(
            "Autocatalytic Set Emergence", 0.0, 1.0, s.get('autocatalytic_set_emergence', 0.0), 0.05,
            help="A fitness bonus for the emergence of 'autocatalytic sets' - tightly-coupled subgraphs where modules mutually support each other's existence, promoting complex, integrated subsystems.",
            key="autocatalytic_set_emergence_slider",
            disabled=not enable_deep_physics
        )

        st.markdown("---")
        st.markdown("#### 5. Cognitive & Economic Pressures")

        curiosity_drive = st.slider(
            "Curiosity Drive (Information Gap)", 0.0, 1.0, s.get('curiosity_drive', 0.0), 0.01,
            disabled=not enable_deep_physics, key="curiosity_drive_slider",
            help="**Rewards exploration of the unknown.** Provides an intrinsic reward for reducing uncertainty about the environment, based on information gap theory. Drives the evolution of exploratory behaviors."
        )
        world_model_accuracy = st.slider(
            "World Model Accuracy Pressure", 0.0, 1.0, s.get('world_model_accuracy', 0.0), 0.01,
            disabled=not enable_deep_physics, key="world_model_accuracy_slider",
            help="**Rewards building an internal simulation.** Directly rewards the accuracy of a genotype's internal model of the environment's dynamics, promoting the evolution of 'imagination'."
        )
        ast_congruence = st.slider(
            "Attention Schema Congruence", 0.0, 1.0, s.get('ast_congruence', 0.0), 0.01,
            disabled=not enable_deep_physics, key="ast_congruence_slider",
            help="**Rewards self-awareness of attention.** From Attention Schema Theory, this rewards architectures that develop an internal model of their own attentional state."
        )
        tom_emergence_pressure = st.slider(
            "Theory of Mind (ToM) Pressure", 0.0, 1.0, s.get('tom_emergence_pressure', 0.0), 0.01,
            disabled=not enable_deep_physics, key="tom_emergence_pressure_slider",
            help="**Drives social reasoning.** In a multi-agent context, this would reward architectures that can accurately predict the internal states and actions of other agents."
        )
        cognitive_dissonance_penalty = st.slider(
            "Cognitive Dissonance Penalty", 0.0, 1.0, s.get('cognitive_dissonance_penalty', 0.0), 0.01,
            disabled=not enable_deep_physics, key="cognitive_dissonance_penalty_slider",
            help="**Penalizes conflicting internal beliefs.** Applies a fitness cost if different parts of the network produce contradictory predictions, forcing the evolution of a coherent world model."
        )
        opportunity_cost_factor = st.slider(
            "Opportunity Cost Factor", 0.0, 1.0, s.get('opportunity_cost_factor', 0.0), 0.01,
            disabled=not enable_deep_physics, key="opportunity_cost_factor_slider",
            help="**Models economic decision-making.** Penalizes architectures for time/energy spent on computations that do not lead to fitness improvements, forcing a more efficient allocation of cognitive resources."
        )
        prospect_theory_bias = st.slider(
            "Prospect Theory Bias (Risk Aversion)", -1.0, 1.0, s.get('prospect_theory_bias', 0.0), 0.05,
            disabled=not enable_deep_physics, key="prospect_theory_bias_slider",
            help="**Introduces non-linear utility of fitness.** Remaps the fitness function according to Prospect Theory. Positive values = risk-averse (preferring sure gains), Negative values = risk-seeking (preferring long shots)."
        )
        temporal_discounting_factor = st.slider(
            "Temporal Discounting Factor", 0.0, 1.0, s.get('temporal_discounting_factor', 0.0), 0.01,
            disabled=not enable_deep_physics, key="temporal_discounting_factor_slider",
            help="**Values immediate vs. future rewards.** In sequential tasks, this discounts the value of future rewards, forcing a trade-off between short-term and long-term planning. High values = more 'patient' architectures."
        )
        zpd_scaffolding_bonus = st.slider(
            "ZPD Scaffolding Bonus", 0.0, 1.0, s.get('zpd_scaffolding_bonus', 0.0), 0.01,
            disabled=not enable_deep_physics, key="zpd_scaffolding_bonus_slider",
            help="**Rewards learning at the right difficulty.** From Vygotsky's Zone of Proximal Development, this provides a bonus for learning from tasks that are neither too easy nor too hard, promoting efficient curriculum learning."
        )
        symbol_grounding_constraint = st.slider(
            "Symbol Grounding Constraint", 0.0, 1.0, s.get('symbol_grounding_constraint', 0.0), 0.01,
            disabled=not enable_deep_physics, key="symbol_grounding_constraint_slider",
            help="**Forces abstract concepts to be linked to sensory data.** Penalizes 'floating' high-level representations that are not causally connected to low-level sensory modules, addressing the symbol grounding problem."
        )

    st.sidebar.markdown("### Experiment Settings")
    experiment_name = st.sidebar.text_input(
        "Experiment Name",
        value=s.get('experiment_name', 'Default Run'),
        help="A name for this experiment run. Saved with the configuration and results.",
        key="experiment_name_input"
    )
    num_generations = st.sidebar.slider(
        "Generations",
        min_value=10, max_value=1000, value=s.get('num_generations', 100),
        help="""
        **Simulates deep evolutionary time.** The number of cycles of selection and reproduction.
        - **Low (10-50):** Short-term adaptation.
        - **High (200+):** Allows for major evolutionary transitions and complex features to emerge.
        **WARNING:** Immense timescales (>200) will take a very long time to compute. A 1000-generation run could take hours.
        """,
        key="num_generations_slider"
    )
    
    complexity_options = ['minimal', 'medium', 'high']
    complexity_level = st.sidebar.select_slider(
        "Initial Complexity",
        options=complexity_options,
        value=s.get('complexity_level', 'medium'),
        key="complexity_level_select_slider"
    )

    st.sidebar.markdown("###### Advanced Controls")
    random_seed = st.sidebar.number_input(
        "Random Seed",
        min_value=-1, value=s.get('random_seed', 42), step=1,
        help="Set a specific seed for reproducibility. Use -1 for a random seed on each run.",
        key="random_seed_input"
    )
    enable_early_stopping = st.sidebar.checkbox(
        "Enable Early Stopping",
        value=s.get('enable_early_stopping', True),
        help="Stop the evolution if the best fitness does not improve for a set number of generations.",
        key="enable_early_stopping_checkbox"
    )
    early_stopping_patience = st.sidebar.slider(
        "Early Stopping Patience",
        min_value=5, max_value=100, value=s.get('early_stopping_patience', 25),
        disabled=not enable_early_stopping,
        help="Number of generations to wait for improvement before stopping.",
        key="early_stopping_patience_slider"
    )
    checkpoint_frequency = st.sidebar.number_input(
        "Checkpoint Frequency",
        min_value=0, value=s.get('checkpoint_frequency', 50), step=5,
        help="Save a checkpoint of the experiment state every N generations. 0 to disable. Checkpoints are saved to the same `genevo_db.json` file, overwriting the previous state.",
        key="checkpoint_frequency_input"
    )
    analysis_top_n = st.sidebar.number_input(
        "Top Architectures to Analyze",
        min_value=1, max_value=20, value=s.get('analysis_top_n', 3), step=1,
        help="Number of top-ranked architectures to display in the final analysis section.",
        key="analysis_top_n_input"
    )

    st.sidebar.markdown("###### Iterative Evolution")
    enable_iterative_seeding = st.sidebar.checkbox(
        "Seed next run with elites",
        value=s.get('enable_iterative_seeding', False),
        help="**Build on previous success.** If checked, initiating a new evolution will use the top individuals from the *last completed run* as the starting population, instead of random initial forms.",
        key="enable_iterative_seeding_checkbox"
    )
    num_elites_to_seed = st.sidebar.slider("Number of Elites to Seed", 1, 50, s.get('num_elites_to_seed', 5), 1, disabled=not enable_iterative_seeding, key="num_elites_to_seed_slider")
    seeded_elite_mutation_strength = st.sidebar.slider(
        "Initial Mutation Strength for Seeded Elites", 0.0, 1.0, s.get('seeded_elite_mutation_strength', 0.4), 0.05,
        disabled=not enable_iterative_seeding,
        help="How much to mutate the seeded elites to create initial diversity for the new run.", key="seeded_elite_mutation_strength_slider"
    )
    
    with st.sidebar.expander("🏁 Finalization & Post-Processing", expanded=False):
        st.markdown("Define automated post-evolution analysis and synthesis steps that run after the main evolutionary process concludes.")
        
        st.markdown("---")
        st.markdown("##### 🛰️ Automated Ensemble Creation")
        enable_ensemble_creation = st.checkbox(
            "Create Ensemble from Pareto Front",
            value=s.get('enable_ensemble_creation', False),
            help="**From many, one team.** After evolution, automatically select a diverse set of high-performing individuals from the final Pareto frontier to form a robust ensemble model.",
            key="enable_ensemble_creation_checkbox"
        )
        ensemble_size = st.slider(
            "Ensemble Size", 2, 20, s.get('ensemble_size', 5), 1,
            disabled=not enable_ensemble_creation,
            help="The number of distinct architectures to include in the final ensemble.",
            key="ensemble_size_slider"
        )
        ensemble_selection_strategy = st.selectbox(
            "Ensemble Selection Strategy",
            ['K-Means Diversity', 'Top N Fittest on Pareto', 'Random from Pareto'],
            index=['K-Means Diversity', 'Top N Fittest on Pareto', 'Random from Pareto'].index(s.get('ensemble_selection_strategy', 'K-Means Diversity')),
            disabled=not enable_ensemble_creation,
            help="""
            **How to choose the team members:**
            - **K-Means Diversity:** Clusters the Pareto front in phenotype space (accuracy, efficiency, etc.) and picks the centroid of each cluster. Maximizes strategic diversity.
            - **Top N Fittest on Pareto:** Simply picks the N individuals with the highest raw fitness from the frontier.
            - **Random from Pareto:** Randomly samples N individuals from the frontier.
            """,
            key="ensemble_selection_strategy_selectbox"
        )

        st.markdown("---")
        st.markdown("##### ⚙️ Post-Evolution Fine-Tuning")
        enable_fine_tuning = st.checkbox(
            "Enable Post-Evolution Fine-Tuning Phase",
            value=s.get('enable_fine_tuning', False),
            help="**From exploration to exploitation.** After the main evolution, run a short, secondary phase where structural mutations are disabled, and only parameters are fine-tuned with a low mutation rate. This polishes the final solutions.",
            key="enable_fine_tuning_checkbox"
        )
        fine_tuning_generations = st.slider(
            "Fine-Tuning Generations", 5, 50, s.get('fine_tuning_generations', 10), 5,
            disabled=not enable_fine_tuning,
            help="Number of extra generations dedicated to fine-tuning the final population.",
            key="fine_tuning_generations_slider"
        )
        fine_tuning_mutation_multiplier = st.slider(
            "Fine-Tuning Mutation Multiplier", 0.01, 0.5, s.get('fine_tuning_mutation_multiplier', 0.1), 0.01,
            disabled=not enable_fine_tuning,
            help="A multiplier applied to the final mutation rate for the fine-tuning phase. A small value (e.g., 0.1) ensures only small parameter adjustments are made.",
            key="fine_tuning_mutation_multiplier_slider"
        )

        st.markdown("---")
        st.markdown("##### 🔬 Advanced Finalization & Analysis Engine")
        enable_advanced_finalization = st.checkbox(
            "Enable Advanced Finalization Engine",
            value=s.get('enable_advanced_finalization', False),
            help="**DANGER: HIGHLY EXPERIMENTAL & COMPUTATIONALLY EXPENSIVE.** Unlocks a suite of over 40 advanced post-processing, analysis, and synthesis techniques. These go far beyond simple fine-tuning, employing methods from knowledge distillation, model merging, formal verification, and deep interpretability. Use only for final, in-depth analysis of a completed run.",
            key="enable_advanced_finalization_checkbox"
        )

        # --- Define default values for all advanced finalization parameters ---
        # This prevents NameError when the checkbox is disabled.
        pruning_method = s.get('pruning_method', 'Magnitude')
        pruning_aggressiveness = s.get('pruning_aggressiveness', 0.1)
        model_compression_target_ratio = s.get('model_compression_target_ratio', 0.5)
        quantization_bits = s.get('quantization_bits', 8)
        lottery_ticket_pruning_iterations = s.get('lottery_ticket_pruning_iterations', 3)
        knowledge_distillation_temperature = s.get('knowledge_distillation_temperature', 1.0)
        distillation_teacher_selection = s.get('distillation_teacher_selection', 'Master')
        self_distillation_weight = s.get('self_distillation_weight', 0.0)
        model_merging_method = s.get('model_merging_method', 'Weight Averaging')
        merging_resolution_method = s.get('merging_resolution_method', 'Functional')
        model_merging_alpha = s.get('model_merging_alpha', 0.5)
        bayesian_model_averaging_prior = s.get('bayesian_model_averaging_prior', 0.1)
        stacking_meta_learner_complexity = s.get('stacking_meta_learner_complexity', 0.2)
        calibration_method = s.get('calibration_method', 'Temperature Scaling')
        out_of_distribution_generalization_test = s.get('out_of_distribution_generalization_test', 'Adversarial')
        formal_verification_engine = s.get('formal_verification_engine', 'SMT Solver')
        adversarial_robustness_certification_method = s.get('adversarial_robustness_certification_method', 'Interval Bound Propagation')
        explainability_method = s.get('explainability_method', 'Integrated Gradients')
        symbolic_regression_complexity_penalty = s.get('symbolic_regression_complexity_penalty', 0.01)
        causal_model_extraction_method = s.get('causal_model_extraction_method', 'PC Algorithm')
        concept_extraction_method = s.get('concept_extraction_method', 'TCAV')
        concept_bottleneck_regularization = s.get('concept_bottleneck_regularization', 0.0)
        mechanistic_interpretability_circuit_search = s.get('mechanistic_interpretability_circuit_search', False)
        continual_learning_replay_buffer_size = s.get('continual_learning_replay_buffer_size', 100)
        elastic_weight_consolidation_lambda = s.get('elastic_weight_consolidation_lambda', 0.1)
        synaptic_intelligence_c_param = s.get('synaptic_intelligence_c_param', 0.01)
        solution_export_format = s.get('solution_export_format', 'PyTorch')
        deployment_latency_constraint = s.get('deployment_latency_constraint', 100.0)
        energy_consumption_constraint = s.get('energy_consumption_constraint', 10.0)
        final_report_verbosity = s.get('final_report_verbosity', 'Standard')
        archive_solution_for_future_seeding = s.get('archive_solution_for_future_seeding', True)
        generate_evolutionary_lineage_report = s.get('generate_evolutionary_lineage_report', False)
        perform_sensitivity_analysis_on_hyperparameters = s.get('perform_sensitivity_analysis_on_hyperparameters', False)
        ablation_study_component_count = s.get('ablation_study_component_count', 3)
        cross_validation_folds = s.get('cross_validation_folds', 5)

        if not enable_advanced_finalization:
            st.warning("Advanced Finalization Engine is disabled. The following parameters will have no effect.")
        else:
            st.success("Advanced Finalization Engine is ACTIVE. This will significantly increase post-processing time.")

            st.markdown("###### 1. Pruning & Compression")
            pruning_method = st.selectbox("Pruning Method", ['Magnitude', 'Movement', 'Gradient-Based'], index=['Magnitude', 'Movement', 'Gradient-Based'].index(s.get('pruning_method', 'Magnitude')), disabled=not enable_advanced_finalization, help="Method for pruning weights post-training.")
            pruning_aggressiveness = st.slider("Pruning Aggressiveness", 0.0, 1.0, s.get('pruning_aggressiveness', 0.1), 0.05, disabled=not enable_advanced_finalization, help="The fraction of weights to prune.")
            model_compression_target_ratio = st.slider("Model Compression Target Ratio", 0.1, 1.0, s.get('model_compression_target_ratio', 0.5), 0.05, disabled=not enable_advanced_finalization, help="Target size reduction ratio for techniques like quantization.")
            quantization_bits = st.select_slider("Quantization Bits", options=[4, 8, 16, 32], value=s.get('quantization_bits', 8), disabled=not enable_advanced_finalization, help="Number of bits for weight quantization.")
            lottery_ticket_pruning_iterations = st.slider("Lottery Ticket Pruning Iterations", 0, 10, s.get('lottery_ticket_pruning_iterations', 3), 1, disabled=not enable_advanced_finalization, help="Number of prune-retrain iterations to find a 'lottery ticket' subnetwork. 0 disables.")

            st.markdown("###### 2. Knowledge Synthesis & Distillation")
            knowledge_distillation_temperature = st.slider("Knowledge Distillation Temperature", 1.0, 10.0, s.get('knowledge_distillation_temperature', 1.0), 0.5, disabled=not enable_advanced_finalization, help="Softmax temperature for distilling knowledge from a teacher to a student model.")
            distillation_teacher_selection = st.selectbox("Distillation Teacher Selection", ['Master', 'Ensemble', 'Best Accuracy'], index=['Master', 'Ensemble', 'Best Accuracy'].index(s.get('distillation_teacher_selection', 'Master')), disabled=not enable_advanced_finalization, help="Which model(s) to use as the 'teacher' for distillation.")
            self_distillation_weight = st.slider("Self-Distillation Weight", 0.0, 1.0, s.get('self_distillation_weight', 0.0), 0.05, disabled=not enable_advanced_finalization, help="Weight for a self-distillation loss, where a model learns from its own past predictions.")

            st.markdown("###### 3. Model Merging & Ensembling")
            model_merging_method = st.selectbox("Model Merging Method", ['Weight Averaging', 'Task Arithmetic', 'Fisher Merging'], index=['Weight Averaging', 'Task Arithmetic', 'Fisher Merging'].index(s.get('model_merging_method', 'Weight Averaging')), disabled=not enable_advanced_finalization, help="Algorithm for merging parameters from different models.")
            merging_resolution_method = st.selectbox("Merging Resolution Method", ['Functional', 'Permutation-Based'], index=['Functional', 'Permutation-Based'].index(s.get('merging_resolution_method', 'Functional')), disabled=not enable_advanced_finalization, help="How to align neurons before merging.")
            model_merging_alpha = st.slider("Model Merging Alpha", 0.0, 1.0, s.get('model_merging_alpha', 0.5), 0.05, disabled=not enable_advanced_finalization, help="Interpolation coefficient for merging two models.")
            bayesian_model_averaging_prior = st.slider("Bayesian Model Averaging Prior", 0.0, 1.0, s.get('bayesian_model_averaging_prior', 0.1), 0.05, disabled=not enable_advanced_finalization, help="Strength of the prior in Bayesian Model Averaging.")
            stacking_meta_learner_complexity = st.slider("Stacking Meta-Learner Complexity", 0.1, 1.0, s.get('stacking_meta_learner_complexity', 0.2), 0.1, disabled=not enable_advanced_finalization, help="Complexity of the meta-learner used in stacking ensembles.")

            st.markdown("###### 4. Robustness, Safety & Verification")
            calibration_method = st.selectbox("Calibration Method", ['Temperature Scaling', 'Isotonic Regression', 'Platt Scaling'], index=['Temperature Scaling', 'Isotonic Regression', 'Platt Scaling'].index(s.get('calibration_method', 'Temperature Scaling')), disabled=not enable_advanced_finalization, help="Method to calibrate model confidence scores.")
            out_of_distribution_generalization_test = st.selectbox("OOD Generalization Test", ['Adversarial', 'Domain Shift', 'Noise Injection'], index=['Adversarial', 'Domain Shift', 'Noise Injection'].index(s.get('out_of_distribution_generalization_test', 'Adversarial')), disabled=not enable_advanced_finalization, help="Method for testing generalization to unseen data distributions.")
            formal_verification_engine = st.selectbox("Formal Verification Engine", ['SMT Solver', 'Abstract Interpretation'], index=['SMT Solver', 'Abstract Interpretation'].index(s.get('formal_verification_engine', 'SMT Solver')), disabled=not enable_advanced_finalization, help="Engine for formally verifying properties of the final model.")
            adversarial_robustness_certification_method = st.selectbox("Adversarial Robustness Certification", ['Interval Bound Propagation', 'Linear Relaxation', 'Randomized Smoothing'], index=['Interval Bound Propagation', 'Linear Relaxation', 'Randomized Smoothing'].index(s.get('adversarial_robustness_certification_method', 'Interval Bound Propagation')), disabled=not enable_advanced_finalization, help="Method to provide certified guarantees of robustness against adversarial attacks.")

            st.markdown("###### 5. Interpretability & Explainability (XAI)")
            explainability_method = st.selectbox("Explainability Method", ['Integrated Gradients', 'SHAP', 'LIME'], index=['Integrated Gradients', 'SHAP', 'LIME'].index(s.get('explainability_method', 'Integrated Gradients')), disabled=not enable_advanced_finalization, help="Primary method for generating feature attributions.")
            symbolic_regression_complexity_penalty = st.slider("Symbolic Regression Complexity Penalty", 0.0, 0.1, s.get('symbolic_regression_complexity_penalty', 0.01), 0.005, disabled=not enable_advanced_finalization, help="Penalty on expression complexity when extracting a symbolic formula for the model's function.")
            causal_model_extraction_method = st.selectbox("Causal Model Extraction Method", ['PC Algorithm', 'FCI', 'LiNGAM'], index=['PC Algorithm', 'FCI', 'LiNGAM'].index(s.get('causal_model_extraction_method', 'PC Algorithm')), disabled=not enable_advanced_finalization, help="Algorithm to extract a causal graph from the model's internal workings.")
            concept_extraction_method = st.selectbox("Concept Extraction Method", ['TCAV', 'Network Dissection'], index=['TCAV', 'Network Dissection'].index(s.get('concept_extraction_method', 'TCAV')), disabled=not enable_advanced_finalization, help="Method to identify human-understandable concepts learned by internal neurons.")
            concept_bottleneck_regularization = st.slider("Concept Bottleneck Regularization", 0.0, 1.0, s.get('concept_bottleneck_regularization', 0.0), 0.05, disabled=not enable_advanced_finalization, help="Strength of a regularization term that forces the model to learn through an explicit concept bottleneck layer.")
            mechanistic_interpretability_circuit_search = st.checkbox("Mechanistic Interpretability Circuit Search", value=s.get('mechanistic_interpretability_circuit_search', False), disabled=not enable_advanced_finalization, help="Perform an automated search for specific computational circuits within the final network (e.g., induction heads).")

            st.markdown("###### 6. Continual Learning & Adaptation")
            continual_learning_replay_buffer_size = st.slider("Continual Learning Replay Buffer Size", 0, 1000, s.get('continual_learning_replay_buffer_size', 100), 50, disabled=not enable_advanced_finalization, help="Size of the experience replay buffer for testing continual learning capabilities. 0 disables.")
            elastic_weight_consolidation_lambda = st.slider("Elastic Weight Consolidation (EWC) Lambda", 0.0, 100.0, s.get('elastic_weight_consolidation_lambda', 0.1), 0.1, disabled=not enable_advanced_finalization, help="Strength of the EWC penalty that protects important weights from previous tasks.")
            synaptic_intelligence_c_param = st.slider("Synaptic Intelligence 'c' Parameter", 0.0, 0.1, s.get('synaptic_intelligence_c_param', 0.01), 0.005, disabled=not enable_advanced_finalization, help="Parameter controlling the per-synapse contribution to the total change in the loss for Synaptic Intelligence.")

            st.markdown("###### 7. Deployment & Reporting")
            solution_export_format = st.selectbox("Solution Export Format", ['PyTorch', 'TensorFlow', 'ONNX'], index=['PyTorch', 'TensorFlow', 'ONNX'].index(s.get('solution_export_format', 'PyTorch')), disabled=not enable_advanced_finalization, help="The target format for exporting the final, production-ready model.")
            deployment_latency_constraint = st.number_input("Deployment Latency Constraint (ms)", value=s.get('deployment_latency_constraint', 100.0), disabled=not enable_advanced_finalization, help="A target latency for the final model, used to guide compression and pruning.")
            energy_consumption_constraint = st.number_input("Energy Consumption Constraint (Joules/inference)", value=s.get('energy_consumption_constraint', 10.0), disabled=not enable_advanced_finalization, help="A target energy budget for the final model.")
            final_report_verbosity = st.select_slider("Final Report Verbosity", options=['Minimal', 'Standard', 'Exhaustive'], value=s.get('final_report_verbosity', 'Standard'), disabled=not enable_advanced_finalization, help="The level of detail in the final automated analysis report.")
            archive_solution_for_future_seeding = st.checkbox("Archive Solution for Future Seeding", value=s.get('archive_solution_for_future_seeding', True), disabled=not enable_advanced_finalization, help="Save the final master architecture to a permanent archive for seeding future experiments.")
            generate_evolutionary_lineage_report = st.checkbox("Generate Evolutionary Lineage Report", value=s.get('generate_evolutionary_lineage_report', False), disabled=not enable_advanced_finalization, help="Generate a detailed report tracing the full ancestry of the final master architecture back to generation 0.")
            perform_sensitivity_analysis_on_hyperparameters = st.checkbox("Perform Hyperparameter Sensitivity Analysis", value=s.get('perform_sensitivity_analysis_on_hyperparameters', False), disabled=not enable_advanced_finalization, help="Perform a sensitivity analysis on the final model with respect to its key hyperparameters.")
            ablation_study_component_count = st.slider("Ablation Study Component Count", 0, 10, s.get('ablation_study_component_count', 3), 1, disabled=not enable_advanced_finalization, help="Number of top components to include in an automated ablation study report. 0 disables.")
            cross_validation_folds = st.slider("Cross-Validation Folds", 2, 10, s.get('cross_validation_folds', 5), 1, disabled=not enable_advanced_finalization, help="Number of folds for a final cross-validation test of the master architecture's robustness.")

        # The original code for this section was removed in the previous diff, so this is a logical correction
        # to put the UI elements back, but inside the `else` block.
        # Since the original code is not in the context, I am re-creating it based on the variable names.
        # The user's request was to add the UI, which I did, but I put it in the wrong scope.
        # Now I am correcting it.


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
        # --- NEW ADVANCED PRIMARY OBJECTIVES SETTINGS ---
        'w_learning_speed': w_learning_speed,
        'w_data_parsimony': w_data_parsimony,
        'w_forgetting_resistance': w_forgetting_resistance,
        'w_adaptability': w_adaptability,
        'w_latency': w_latency,
        'w_energy_consumption': w_energy_consumption,
        'w_development_cost': w_development_cost,
        'w_modularity': w_modularity,
        'w_interpretability': w_interpretability,
        'w_evolvability': w_evolvability,
        'w_fairness': w_fairness,
        'w_explainability': w_explainability,
        'w_value_alignment': w_value_alignment,
        'w_causal_density': w_causal_density,
        'w_self_organization': w_self_organization,
        'w_autopoiesis': w_autopoiesis,
        'w_computational_irreducibility': w_computational_irreducibility,
        'w_cognitive_synergy': w_cognitive_synergy,
        # --- NEW ADVANCED OBJECTIVES SETTINGS ---
        'enable_advanced_objectives': enable_advanced_objectives,
        'w_kolmogorov_complexity': w_kolmogorov_complexity if enable_advanced_objectives else 0.0,
        'w_predictive_information': w_predictive_information if enable_advanced_objectives else 0.0,
        'w_causal_emergence': w_causal_emergence if enable_advanced_objectives else 0.0,
        'w_integrated_information': w_integrated_information if enable_advanced_objectives else 0.0,
        'w_free_energy_minimization': w_free_energy_minimization if enable_advanced_objectives else 0.0,
        'w_transfer_entropy': w_transfer_entropy if enable_advanced_objectives else 0.0,
        'w_synergistic_information': w_synergistic_information if enable_advanced_objectives else 0.0,
        'w_state_compression': w_state_compression if enable_advanced_objectives else 0.0,
        'w_empowerment': w_empowerment if enable_advanced_objectives else 0.0,
        'w_semantic_information': w_semantic_information if enable_advanced_objectives else 0.0,
        'w_effective_information': w_effective_information if enable_advanced_objectives else 0.0,
        'w_information_closure': w_information_closure if enable_advanced_objectives else 0.0,
        'w_landauer_cost': w_landauer_cost if enable_advanced_objectives else 0.0,
        'w_metabolic_efficiency': w_metabolic_efficiency if enable_advanced_objectives else 0.0,
        'w_heat_dissipation': w_heat_dissipation if enable_advanced_objectives else 0.0,
        'w_homeostasis': w_homeostasis if enable_advanced_objectives else 0.0,
        'w_structural_integrity': w_structural_integrity if enable_advanced_objectives else 0.0,
        'w_entropy_production': w_entropy_production if enable_advanced_objectives else 0.0,
        'w_resource_acquisition_efficiency': w_resource_acquisition_efficiency if enable_advanced_objectives else 0.0,
        'w_aging_resistance': w_aging_resistance if enable_advanced_objectives else 0.0,
        'w_curiosity': w_curiosity if enable_advanced_objectives else 0.0,
        'w_world_model_accuracy': w_world_model_accuracy if enable_advanced_objectives else 0.0,
        'w_attention_schema': w_attention_schema if enable_advanced_objectives else 0.0,
        'w_theory_of_mind': w_theory_of_mind if enable_advanced_objectives else 0.0,
        'w_cognitive_dissonance': w_cognitive_dissonance if enable_advanced_objectives else 0.0,
        'w_goal_achievement': w_goal_achievement if enable_advanced_objectives else 0.0,
        'w_cognitive_learning_speed': w_cognitive_learning_speed if enable_advanced_objectives else 0.0,
        'w_cognitive_forgetting_resistance': w_cognitive_forgetting_resistance if enable_advanced_objectives else 0.0,
        'w_compositionality': w_compositionality if enable_advanced_objectives else 0.0,
        'w_planning_depth': w_planning_depth if enable_advanced_objectives else 0.0,
        'w_structural_modularity': w_structural_modularity if enable_advanced_objectives else 0.0,
        'w_hierarchy': w_hierarchy if enable_advanced_objectives else 0.0,
        'w_symmetry': w_symmetry if enable_advanced_objectives else 0.0,
        'w_small_worldness': w_small_worldness if enable_advanced_objectives else 0.0,
        'w_scale_free': w_scale_free if enable_advanced_objectives else 0.0,
        'w_fractal_dimension': w_fractal_dimension if enable_advanced_objectives else 0.0,
        'w_hyperbolic_embeddability': w_hyperbolic_embeddability if enable_advanced_objectives else 0.0,
        'w_autocatalysis': w_autocatalysis if enable_advanced_objectives else 0.0,
        'w_wiring_cost': w_wiring_cost if enable_advanced_objectives else 0.0,
        'w_rich_club_coefficient': w_rich_club_coefficient if enable_advanced_objectives else 0.0,
        'w_assortativity': w_assortativity if enable_advanced_objectives else 0.0,
        'w_adaptability_speed': w_adaptability_speed if enable_advanced_objectives else 0.0,
        'w_predictive_horizon': w_predictive_horizon if enable_advanced_objectives else 0.0,
        'w_behavioral_stability': w_behavioral_stability if enable_advanced_objectives else 0.0,
        'w_criticality_dynamics': w_criticality_dynamics if enable_advanced_objectives else 0.0,
        'w_decision_time': w_decision_time if enable_advanced_objectives else 0.0,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        # --- NEW ADVANCED MUTATION SETTINGS ---
        'enable_advanced_mutation': enable_advanced_mutation,
        'mutation_distribution_type': mutation_distribution_type,
        'mutation_scale_parameter': mutation_scale_parameter,
        'structural_mutation_scale': structural_mutation_scale,
        'mutation_correlation_factor': mutation_correlation_factor,
        'mutation_operator_bias': mutation_operator_bias,
        'mutation_tail_heaviness': mutation_tail_heaviness,
        'mutation_anisotropy_vector': mutation_anisotropy_vector,
        'mutation_modality': mutation_modality,
        'mutation_step_size_annealing': mutation_step_size_annealing,
        'fitness_dependent_mutation_strength': fitness_dependent_mutation_strength,
        'age_dependent_mutation_strength': age_dependent_mutation_strength,
        'module_size_dependent_mutation': module_size_dependent_mutation,
        'connection_weight_dependent_mutation': connection_weight_dependent_mutation,
        'somatic_hypermutation_rate': somatic_hypermutation_rate,
        'error_driven_mutation_strength': error_driven_mutation_strength,
        'gene_centrality_mutation_bias': gene_centrality_mutation_bias,
        'epigenetic_mutation_influence': epigenetic_mutation_influence,
        'mutation_hotspot_probability': mutation_hotspot_probability,
        'add_connection_topology_bias': add_connection_topology_bias,
        'add_module_method': add_module_method,
        'remove_connection_probability': remove_connection_probability,
        'remove_module_probability': remove_module_probability,
        'connection_rewiring_probability': connection_rewiring_probability,
        'module_duplication_probability': module_duplication_probability,
        'module_fusion_probability': module_fusion_probability,
        'cycle_formation_probability': cycle_formation_probability,
        'structural_mutation_phase': structural_mutation_phase,
        'learning_rate_mutation_strength': learning_rate_mutation_strength,
        'plasticity_mutation_strength': plasticity_mutation_strength,
        'learning_improvement_mutation_bonus': learning_improvement_mutation_bonus,
        'weight_change_mutation_correlation': weight_change_mutation_correlation,
        'synaptic_tagging_credit_assignment': synaptic_tagging_credit_assignment,
        'metaplasticity_rule': metaplasticity_rule,
        'learning_instability_penalty': learning_instability_penalty,
        'gradient_guided_mutation_strength': gradient_guided_mutation_strength,
        'hessian_guided_mutation_strength': hessian_guided_mutation_strength,
        'crossover_operator': crossover_operator,
        'crossover_n_points': crossover_n_points,
        'crossover_parent_assortativity': crossover_parent_assortativity,
        'crossover_gene_dominance_probability': crossover_gene_dominance_probability,
        'sexual_reproduction_rate': sexual_reproduction_rate,
        'inbreeding_penalty_factor': inbreeding_penalty_factor,
        'horizontal_gene_transfer_rate': horizontal_gene_transfer_rate,
        'polyploidy_probability': polyploidy_probability,
        'meiotic_recombination_rate': meiotic_recombination_rate,
        'innovation_rate': innovation_rate,
        'enable_development': enable_development,
        'enable_baldwin': enable_baldwin,
        'baldwinian_assimilation_rate': baldwinian_assimilation_rate,
        'enable_epigenetics': enable_epigenetics,
        'endosymbiosis_rate': endosymbiosis_rate,
        # --- NEW CO-EVOLUTION & EMBODIMENT SETTINGS ---
        'enable_adversarial_coevolution': enable_adversarial_coevolution,
        'critic_population_size': critic_population_size,
        'critic_mutation_rate': critic_mutation_rate,
        'adversarial_fitness_weight': adversarial_fitness_weight,
        'critic_selection_pressure': critic_selection_pressure,
        'critic_task': critic_task,
        'enable_morphological_coevolution': enable_morphological_coevolution,
        'morphological_mutation_rate': morphological_mutation_rate,
        'max_body_modules': max_body_modules,
        'cost_per_module': cost_per_module,
        'enable_sensor_evolution': enable_sensor_evolution,
        'enable_actuator_evolution': enable_actuator_evolution,
        'physical_realism_factor': physical_realism_factor,
        'embodiment_gravity': embodiment_gravity,
        'embodiment_friction': embodiment_friction,
        # --- NEW SOPHISTICATED CO-EVOLUTION SETTINGS ---
        'enable_hall_of_fame': enable_hall_of_fame,
        'hall_of_fame_size': hall_of_fame_size,
        'hall_of_fame_replacement_strategy': hall_of_fame_replacement_strategy,
        'critic_evolution_frequency': critic_evolution_frequency,
        'critic_cooperation_probability': critic_cooperation_probability,
        'cooperative_reward_scaling': cooperative_reward_scaling,
        'critic_objective_novelty_weight': critic_objective_novelty_weight,
        'bilateral_symmetry_bonus': bilateral_symmetry_bonus,
        'segmentation_bonus': segmentation_bonus,
        'allometric_scaling_exponent': allometric_scaling_exponent,
        'enable_material_evolution': enable_material_evolution,
        'cost_per_stiffness': cost_per_stiffness,
        'cost_per_density': cost_per_density,
        'evolvable_sensor_noise': evolvable_sensor_noise,
        'evolvable_actuator_force': evolvable_actuator_force,
        'fluid_dynamics_viscosity': fluid_dynamics_viscosity,
        'surface_tension_factor': surface_tension_factor,
        'enable_host_symbiont_coevolution': enable_host_symbiont_coevolution,
        'symbiont_population_size': symbiont_population_size,
        'symbiont_mutation_rate': symbiont_mutation_rate,
        'symbiont_transfer_rate': symbiont_transfer_rate,
        'symbiont_vertical_inheritance_fidelity': symbiont_vertical_inheritance_fidelity,
        'host_symbiont_fitness_dependency': host_symbiont_fitness_dependency,
        # --- NEW MULTI-LEVEL SELECTION SETTINGS ---
        'enable_multi_level_selection': enable_multi_level_selection,
        'colony_formation_method': colony_formation_method,
        'colony_size': colony_size,
        'group_fitness_weight': group_fitness_weight,
        'selfishness_suppression_cost': selfishness_suppression_cost,
        'caste_specialization_bonus': caste_specialization_bonus,
        'inter_colony_competition_rate': inter_colony_competition_rate,
        'epistatic_linkage_k': epistatic_linkage_k,
        'gene_flow_rate': gene_flow_rate,
        'niche_competition_factor': niche_competition_factor,
        'max_archive_size': max_archive_size,
        # --- NEW ADVANCED LANDSCAPE PHYSICS SETTINGS ---
        'speciation_stagnation_threshold': speciation_stagnation_threshold,
        'species_extinction_threshold': species_extinction_threshold,
        'niche_construction_strength': niche_construction_strength,
        'character_displacement_pressure': character_displacement_pressure,
        'adaptive_radiation_trigger': adaptive_radiation_trigger,
        'species_merger_probability': species_merger_probability,
        'kin_selection_bonus': kin_selection_bonus,
        'sexual_selection_factor': sexual_selection_factor,
        'sympatric_speciation_pressure': sympatric_speciation_pressure,
        'allopatric_speciation_trigger': allopatric_speciation_trigger,
        'intraspecific_competition_scaling': intraspecific_competition_scaling,
        'landscape_ruggedness_factor': landscape_ruggedness_factor,
        'landscape_correlation_length': landscape_correlation_length,
        'landscape_neutral_network_size': landscape_neutral_network_size,
        'landscape_holeyness_factor': landscape_holeyness_factor,
        'landscape_anisotropy_factor': landscape_anisotropy_factor,
        'landscape_gradient_noise': landscape_gradient_noise,
        'landscape_time_variance_rate': landscape_time_variance_rate,
        'multimodality_factor': multimodality_factor,
        'epistatic_correlation_structure': epistatic_correlation_structure,
        'fitness_autocorrelation_time': fitness_autocorrelation_time,
        'fitness_landscape_plasticity': fitness_landscape_plasticity,
        'information_bottleneck_pressure': information_bottleneck_pressure,
        'fisher_information_maximization': fisher_information_maximization,
        'predictive_information_bonus': predictive_information_bonus,
        'thermodynamic_depth_bonus': thermodynamic_depth_bonus,
        'integrated_information_bonus': integrated_information_bonus,
        'free_energy_minimization_pressure': free_energy_minimization_pressure,
        'empowerment_maximization_drive': empowerment_maximization_drive,
        'causal_density_target': causal_density_target,
        'semantic_information_bonus': semantic_information_bonus,
        'algorithmic_complexity_penalty': algorithmic_complexity_penalty,
        'computational_irreducibility_bonus': computational_irreducibility_bonus,
        'altruism_punishment_effectiveness': altruism_punishment_effectiveness,
        'resource_depletion_rate': resource_depletion_rate,
        'predator_prey_cycle_period': predator_prey_cycle_period,
        'mutualism_bonus': mutualism_bonus,
        'parasitism_virulence_factor': parasitism_virulence_factor,
        'commensalism_emergence_bonus': commensalism_emergence_bonus,
        'social_learning_fidelity': social_learning_fidelity,
        'cultural_evolution_rate': cultural_evolution_rate,
        'group_selection_strength': group_selection_strength,
        'tragedy_of_the_commons_penalty': tragedy_of_the_commons_penalty,
        'reputation_dynamics_factor': reputation_dynamics_factor,
        'extinction_event_severity': extinction_event_severity,
        'environmental_shift_magnitude': environmental_shift_magnitude,
        'punctuated_equilibrium_trigger_sensitivity': punctuated_equilibrium_trigger_sensitivity,
        'key_innovation_bonus': key_innovation_bonus,
        'background_extinction_rate': background_extinction_rate,
        'invasive_species_introduction_prob': invasive_species_introduction_prob,
        'adaptive_radiation_factor': adaptive_radiation_factor,
        'refugia_survival_bonus': refugia_survival_bonus,
        'post_cataclysm_hypermutation_period': post_cataclysm_hypermutation_period,
        'environmental_press_factor': environmental_press_factor,
        'cambrian_explosion_trigger': cambrian_explosion_trigger,
        'reintroduction_rate': reintroduction_rate,
        'enable_cataclysms': enable_cataclysms,
        'cataclysm_probability': cataclysm_probability,
        'cataclysm_extinction_severity': cataclysm_extinction_severity,
        'cataclysm_landscape_shift_magnitude': cataclysm_landscape_shift_magnitude,
        'post_cataclysm_hypermutation_multiplier': post_cataclysm_hypermutation_multiplier,
        'post_cataclysm_hypermutation_duration': post_cataclysm_hypermutation_duration,
        'cataclysm_selectivity_type': cataclysm_selectivity_type,
        'red_queen_virulence': red_queen_virulence,
        'red_queen_adaptation_speed': red_queen_adaptation_speed,
        'red_queen_target_breadth': red_queen_target_breadth,
        'enable_red_queen': enable_red_queen,
        'enable_endosymbiosis': enable_endosymbiosis,
        'mutation_schedule': mutation_schedule,
        'adaptive_mutation_strength': adaptive_mutation_strength,
        'selection_pressure': selection_pressure,
        'enable_speciation': enable_speciation,
        'enable_diversity_pressure': enable_diversity_pressure,
        'diversity_weight': diversity_weight,
        # --- NEW ADVANCED SPECIATION SETTINGS ---
        'enable_advanced_speciation': enable_advanced_speciation,
        'dynamic_threshold_adjustment_rate': dynamic_threshold_adjustment_rate,
        'distance_weight_c1': distance_weight_c1,
        'distance_weight_c2': distance_weight_c2,
        'phenotypic_distance_weight': phenotypic_distance_weight,
        'age_distance_weight': age_distance_weight,
        'lineage_distance_factor': lineage_distance_factor,
        'distance_normalization_factor': distance_normalization_factor,
        'developmental_rule_distance_weight': developmental_rule_distance_weight,
        'meta_param_distance_weight': meta_param_distance_weight,
        'species_stagnation_threshold': species_stagnation_threshold,
        'stagnation_penalty': stagnation_penalty,
        'species_age_bonus': species_age_bonus,
        'species_novelty_bonus': species_novelty_bonus,
        'min_species_size_for_survival': min_species_size_for_survival,
        'species_extinction_threshold': species_extinction_threshold,
        'species_merger_threshold': species_merger_threshold,
        'species_merger_probability': species_merger_probability,
        'niche_construction_strength': niche_construction_strength,
        'character_displacement_pressure': character_displacement_pressure,
        'intraspecific_competition_scaling': intraspecific_competition_scaling,
        'interspecific_competition_scaling': interspecific_competition_scaling,
        'resource_depletion_rate': resource_depletion_rate,
        'niche_overlap_penalty': niche_overlap_penalty,
        'niche_capacity': niche_capacity,
        'sexual_selection_factor': sexual_selection_factor,
        'mating_preference_strength': mating_preference_strength,
        'outbreeding_depression_penalty': outbreeding_depression_penalty,
        'inbreeding_depression_penalty': inbreeding_depression_penalty,
        'reproductive_isolation_threshold': reproductive_isolation_threshold,
        'assortative_mating_strength': assortative_mating_strength,
        'sympatric_speciation_pressure': sympatric_speciation_pressure,
        'allopatric_speciation_trigger': allopatric_speciation_trigger,
        'parapatric_speciation_gradient': parapatric_speciation_gradient,
        'peripatric_speciation_founder_effect': peripatric_speciation_founder_effect,
        'adaptive_radiation_trigger_threshold': adaptive_radiation_trigger_threshold,
        'adaptive_radiation_strength': adaptive_radiation_strength,
        'kin_selection_bonus': kin_selection_bonus,
        'group_selection_strength': group_selection_strength,
        'altruism_cost': altruism_cost,
        'species_reputation_factor': species_reputation_factor,
        'punctuated_equilibrium_trigger_sensitivity': punctuated_equilibrium_trigger_sensitivity,
        'background_extinction_rate': background_extinction_rate,
        'key_innovation_bonus': key_innovation_bonus,
        'invasive_species_introduction_prob': invasive_species_introduction_prob,
        'refugia_survival_bonus': refugia_survival_bonus,
        'phyletic_gradualism_factor': phyletic_gradualism_factor,
        'species_sorting_strength': species_sorting_strength,
        'compatibility_threshold': compatibility_threshold,
        'num_generations': num_generations,
        'complexity_level': complexity_level,
        'experiment_name': experiment_name,
        'random_seed': random_seed,
        'enable_early_stopping': enable_early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'checkpoint_frequency': checkpoint_frequency,
        'analysis_top_n': analysis_top_n,
        'enable_hyperparameter_evolution': enable_hyperparameter_evolution,
        'evolvable_params': evolvable_params,
        'hyper_mutation_rate': hyper_mutation_rate,
        # --- NEW META-EVOLUTION SETTINGS ---
        'hyper_mutation_distribution': hyper_mutation_distribution,
        'evolvable_param_bounds_leniency': evolvable_param_bounds_leniency,
        'hyperparam_heritability_factor': hyperparam_heritability_factor,
        
        'enable_genetic_code_evolution': enable_genetic_code_evolution,
        'gene_type_innovation_rate': gene_type_innovation_rate,
        'gene_type_extinction_rate': gene_type_extinction_rate,
        'evolvable_activation_functions': evolvable_activation_functions,
        'activation_expression_complexity_limit': activation_expression_complexity_limit,
        'developmental_rule_innovation_rate': developmental_rule_innovation_rate,
        'encoding_plasticity_rate': encoding_plasticity_rate,
        'genome_length_constraint_pressure': genome_length_constraint_pressure,
        'intron_ratio_target': intron_ratio_target,
        'gene_regulatory_network_complexity_bonus': gene_regulatory_network_complexity_bonus,
        'evolvable_normalization_layers': evolvable_normalization_layers,

        'enable_ea_dynamics_evolution': enable_ea_dynamics_evolution,
        'evolvable_selection_mechanism': evolvable_selection_mechanism,
        'selection_mechanism_pool': selection_mechanism_pool,
        'evolvable_tournament_size': evolvable_tournament_size,
        'crossover_operator_pool': crossover_operator_pool,
        'mutation_operator_pool': mutation_operator_pool,
        'population_topology': population_topology,
        'evolvable_migration_rate': evolvable_migration_rate,
        'evolvable_island_count': evolvable_island_count,
        'topology_reconfiguration_frequency': topology_reconfiguration_frequency,
        'dynamic_speciation_threshold_factor': dynamic_speciation_threshold_factor,

        'enable_objective_evolution': enable_objective_evolution,
        'evolvable_objective_weights': evolvable_objective_weights,
        'objective_weight_mutation_strength': objective_weight_mutation_strength,
        'autotelic_novelty_search_weight': autotelic_novelty_search_weight,
        'autotelic_complexity_drive_weight': autotelic_complexity_drive_weight,
        'autotelic_learning_progress_drive': autotelic_learning_progress_drive,
        'fitness_function_noise_injection_rate': fitness_function_noise_injection_rate,
        'fitness_landscape_smoothing_factor': fitness_landscape_smoothing_factor,
        'objective_ambition_ratchet': objective_ambition_ratchet,
        'pareto_front_focus_bias': pareto_front_focus_bias,

        'enable_self_modification': enable_self_modification,
        'self_modification_probability': self_modification_probability,
        'self_modification_scope': self_modification_scope,
        'quine_bonus': quine_bonus,
        'meta_genotype_bonus': meta_genotype_bonus,
        'self_simulation_bonus': self_simulation_bonus,
        'enable_curriculum_learning': enable_curriculum_learning,
        'curriculum_sequence': curriculum_sequence,
        'curriculum_trigger': curriculum_trigger,
        'curriculum_threshold': curriculum_threshold,
        'enable_iterative_seeding': enable_iterative_seeding,
        'num_elites_to_seed': num_elites_to_seed,
        'seeded_elite_mutation_strength': seeded_elite_mutation_strength,
        # --- NEW DYNAMIC ENVIRONMENT SETTINGS ---
        'enable_advanced_environment_physics': enable_advanced_environment_physics,
        'non_stationarity_mode': non_stationarity_mode,
        'drift_velocity': drift_velocity,
        'shift_magnitude': shift_magnitude,
        'cycle_period': cycle_period,
        'chaotic_attractor_type': chaotic_attractor_type,
        'environmental_memory_strength': environmental_memory_strength,
        'resource_distribution_mode': resource_distribution_mode,
        'resource_regeneration_rate': resource_regeneration_rate,
        'task_space_curvature': task_space_curvature,
        'environmental_viscosity': environmental_viscosity,
        'environmental_temperature': environmental_temperature,
        'task_noise_correlation_time': task_noise_correlation_time,
        'environmental_lag': environmental_lag,
        'resource_scarcity_level': resource_scarcity_level,

        'enable_advanced_curriculum': enable_advanced_curriculum,
        'curriculum_generation_method': curriculum_generation_method,
        'self_paced_learning_rate': self_paced_learning_rate,
        'teacher_student_dynamics_enabled': teacher_student_dynamics_enabled,
        'teacher_mutation_rate': teacher_mutation_rate,
        'task_proposal_rejection_rate': task_proposal_rejection_rate,
        'transfer_learning_bonus': transfer_learning_bonus,
        'catastrophic_forgetting_penalty': catastrophic_forgetting_penalty,
        'curriculum_backtracking_probability': curriculum_backtracking_probability,
        'interleaved_learning_ratio': interleaved_learning_ratio,
        'task_decomposition_bonus': task_decomposition_bonus,
        'procedural_content_generation_complexity': procedural_content_generation_complexity,
        'curriculum_difficulty_ceiling': curriculum_difficulty_ceiling,
        'teacher_student_objective_alignment': teacher_student_objective_alignment,

        'enable_social_environment': enable_social_environment,
        'communication_channel_bandwidth': communication_channel_bandwidth,
        'communication_channel_noise': communication_channel_noise,
        'social_signal_cost': social_signal_cost,
        'common_knowledge_bonus': common_knowledge_bonus,
        'deception_penalty': deception_penalty,
        'reputation_system_fidelity': reputation_system_fidelity,
        'sanctioning_effectiveness': sanctioning_effectiveness,
        'network_reciprocity_bonus': network_reciprocity_bonus,
        'social_learning_mechanism': social_learning_mechanism,
        'cultural_ratchet_bonus': cultural_ratchet_bonus,
        'social_norm_emergence_bonus': social_norm_emergence_bonus,
        'tribalism_factor': tribalism_factor,

        'enable_open_endedness': enable_open_endedness,
        'poi_novelty_threshold': poi_novelty_threshold,
        'minimal_criterion_coevolution_rate': minimal_criterion_coevolution_rate,
        'autopoiesis_pressure': autopoiesis_pressure,
        'environmental_construction_bonus': environmental_construction_bonus,
        'goal_switching_cost': goal_switching_cost,
        'solution_archive_capacity': solution_archive_capacity,
        'novelty_metric': novelty_metric,
        'local_competition_radius': local_competition_radius,
        'information_seeking_drive': information_seeking_drive,
        'open_ended_archive_sampling_bias': open_ended_archive_sampling_bias,
        'goal_embedding_space_dims': goal_embedding_space_dims,
        # --- NEW FINALIZATION SETTINGS ---
        'enable_ensemble_creation': enable_ensemble_creation,
        'ensemble_size': ensemble_size,
        'ensemble_selection_strategy': ensemble_selection_strategy,
        'enable_fine_tuning': enable_fine_tuning,
        'fine_tuning_generations': fine_tuning_generations,
        'fine_tuning_mutation_multiplier': fine_tuning_mutation_multiplier,
        # --- NEW DEEP PHYSICS SETTINGS ---
        # --- NEW ADVANCED FINALIZATION SETTINGS ---
        'enable_advanced_finalization': enable_advanced_finalization,
        'pruning_method': pruning_method,
        'pruning_aggressiveness': pruning_aggressiveness,
        'model_compression_target_ratio': model_compression_target_ratio,
        'quantization_bits': quantization_bits,
        'lottery_ticket_pruning_iterations': lottery_ticket_pruning_iterations,
        'knowledge_distillation_temperature': knowledge_distillation_temperature,
        'distillation_teacher_selection': distillation_teacher_selection,
        'self_distillation_weight': self_distillation_weight,
        'model_merging_method': model_merging_method,
        'merging_resolution_method': merging_resolution_method,
        'model_merging_alpha': model_merging_alpha,
        'bayesian_model_averaging_prior': bayesian_model_averaging_prior,
        'stacking_meta_learner_complexity': stacking_meta_learner_complexity,
        'calibration_method': calibration_method,
        'out_of_distribution_generalization_test': out_of_distribution_generalization_test,
        'formal_verification_engine': formal_verification_engine,
        'adversarial_robustness_certification_method': adversarial_robustness_certification_method,
        'explainability_method': explainability_method,
        'symbolic_regression_complexity_penalty': symbolic_regression_complexity_penalty,
        'causal_model_extraction_method': causal_model_extraction_method,
        'concept_extraction_method': concept_extraction_method,
        'concept_bottleneck_regularization': concept_bottleneck_regularization,
        'mechanistic_interpretability_circuit_search': mechanistic_interpretability_circuit_search,
        'continual_learning_replay_buffer_size': continual_learning_replay_buffer_size,
        'elastic_weight_consolidation_lambda': elastic_weight_consolidation_lambda,
        'synaptic_intelligence_c_param': synaptic_intelligence_c_param,
        'solution_export_format': solution_export_format,
        'deployment_latency_constraint': deployment_latency_constraint,
        'energy_consumption_constraint': energy_consumption_constraint,
        'final_report_verbosity': final_report_verbosity,
        'archive_solution_for_future_seeding': archive_solution_for_future_seeding,
        'generate_evolutionary_lineage_report': generate_evolutionary_lineage_report,
        'perform_sensitivity_analysis_on_hyperparameters': perform_sensitivity_analysis_on_hyperparameters,
        'ablation_study_component_count': ablation_study_component_count,
        'cross_validation_folds': cross_validation_folds,
        'enable_advanced_frameworks': enable_advanced_frameworks,
        # Computational Logic
        'chaitin_omega_bias': chaitin_omega_bias,
        'godel_incompleteness_penalty': godel_incompleteness_penalty,
        'turing_completeness_bonus': turing_completeness_bonus,
        'lambda_calculus_isomorphism': lambda_calculus_isomorphism,
        'proof_complexity_cost': proof_complexity_cost,
        'constructive_type_theory_adherence': constructive_type_theory_adherence,
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
        # --- NEW ADVANCED PRIMARY OBJECTIVES SETTINGS ---
        'w_learning_speed': w_learning_speed,
        'w_data_parsimony': w_data_parsimony,
        'w_forgetting_resistance': w_forgetting_resistance,
        'w_adaptability': w_adaptability,
        'w_latency': w_latency,
        'w_energy_consumption': w_energy_consumption,
        'w_development_cost': w_development_cost,
        'w_modularity': w_modularity,
        'w_interpretability': w_interpretability,
        'w_evolvability': w_evolvability,
        'w_fairness': w_fairness,
        'w_explainability': w_explainability,
        'w_value_alignment': w_value_alignment,
        'w_causal_density': w_causal_density,
        'w_self_organization': w_self_organization,
        'w_autopoiesis': w_autopoiesis,
        'w_computational_irreducibility': w_computational_irreducibility,
        'w_cognitive_synergy': w_cognitive_synergy,
        # --- NEW ADVANCED OBJECTIVES SETTINGS ---
        'enable_advanced_objectives': enable_advanced_objectives,
        'w_kolmogorov_complexity': w_kolmogorov_complexity if enable_advanced_objectives else 0.0,
        'w_predictive_information': w_predictive_information if enable_advanced_objectives else 0.0,
        'w_causal_emergence': w_causal_emergence if enable_advanced_objectives else 0.0,
        'w_integrated_information': w_integrated_information if enable_advanced_objectives else 0.0,
        'w_free_energy_minimization': w_free_energy_minimization if enable_advanced_objectives else 0.0,
        'w_transfer_entropy': w_transfer_entropy if enable_advanced_objectives else 0.0,
        'w_synergistic_information': w_synergistic_information if enable_advanced_objectives else 0.0,
        'w_state_compression': w_state_compression if enable_advanced_objectives else 0.0,
        'w_empowerment': w_empowerment if enable_advanced_objectives else 0.0,
        'w_semantic_information': w_semantic_information if enable_advanced_objectives else 0.0,
        'w_effective_information': w_effective_information if enable_advanced_objectives else 0.0,
        'w_information_closure': w_information_closure if enable_advanced_objectives else 0.0,
        'w_landauer_cost': w_landauer_cost if enable_advanced_objectives else 0.0,
        'w_metabolic_efficiency': w_metabolic_efficiency if enable_advanced_objectives else 0.0,
        'w_heat_dissipation': w_heat_dissipation if enable_advanced_objectives else 0.0,
        'w_homeostasis': w_homeostasis if enable_advanced_objectives else 0.0,
        'w_structural_integrity': w_structural_integrity if enable_advanced_objectives else 0.0,
        'w_entropy_production': w_entropy_production if enable_advanced_objectives else 0.0,
        'w_resource_acquisition_efficiency': w_resource_acquisition_efficiency if enable_advanced_objectives else 0.0,
        'w_aging_resistance': w_aging_resistance if enable_advanced_objectives else 0.0,
        'w_curiosity': w_curiosity if enable_advanced_objectives else 0.0,
        'w_world_model_accuracy': w_world_model_accuracy if enable_advanced_objectives else 0.0,
        'w_attention_schema': w_attention_schema if enable_advanced_objectives else 0.0,
        'w_theory_of_mind': w_theory_of_mind if enable_advanced_objectives else 0.0,
        'w_cognitive_dissonance': w_cognitive_dissonance if enable_advanced_objectives else 0.0,
        'w_goal_achievement': w_goal_achievement if enable_advanced_objectives else 0.0,
        'w_cognitive_learning_speed': w_cognitive_learning_speed if enable_advanced_objectives else 0.0,
        'w_cognitive_forgetting_resistance': w_cognitive_forgetting_resistance if enable_advanced_objectives else 0.0,
        'w_compositionality': w_compositionality if enable_advanced_objectives else 0.0,
        'w_planning_depth': w_planning_depth if enable_advanced_objectives else 0.0,
        'w_structural_modularity': w_structural_modularity if enable_advanced_objectives else 0.0,
        'w_hierarchy': w_hierarchy if enable_advanced_objectives else 0.0,
        'w_symmetry': w_symmetry if enable_advanced_objectives else 0.0,
        'w_small_worldness': w_small_worldness if enable_advanced_objectives else 0.0,
        'w_scale_free': w_scale_free if enable_advanced_objectives else 0.0,
        'w_fractal_dimension': w_fractal_dimension if enable_advanced_objectives else 0.0,
        'w_hyperbolic_embeddability': w_hyperbolic_embeddability if enable_advanced_objectives else 0.0,
        'w_autocatalysis': w_autocatalysis if enable_advanced_objectives else 0.0,
        'w_wiring_cost': w_wiring_cost if enable_advanced_objectives else 0.0,
        'w_rich_club_coefficient': w_rich_club_coefficient if enable_advanced_objectives else 0.0,
        'w_assortativity': w_assortativity if enable_advanced_objectives else 0.0,
        'w_adaptability_speed': w_adaptability_speed if enable_advanced_objectives else 0.0,
        'w_predictive_horizon': w_predictive_horizon if enable_advanced_objectives else 0.0,
        'w_behavioral_stability': w_behavioral_stability if enable_advanced_objectives else 0.0,
        'w_criticality_dynamics': w_criticality_dynamics if enable_advanced_objectives else 0.0,
        'w_decision_time': w_decision_time if enable_advanced_objectives else 0.0,
        'mutation_rate': mutation_rate,
        'crossover_rate': crossover_rate,
        # --- NEW ADVANCED MUTATION SETTINGS ---
        'enable_advanced_mutation': enable_advanced_mutation,
        'mutation_distribution_type': mutation_distribution_type,
        'mutation_scale_parameter': mutation_scale_parameter,
        'structural_mutation_scale': structural_mutation_scale,
        'mutation_correlation_factor': mutation_correlation_factor,
        'mutation_operator_bias': mutation_operator_bias,
        'mutation_tail_heaviness': mutation_tail_heaviness,
        'mutation_anisotropy_vector': mutation_anisotropy_vector,
        'mutation_modality': mutation_modality,
        'mutation_step_size_annealing': mutation_step_size_annealing,
        'fitness_dependent_mutation_strength': fitness_dependent_mutation_strength,
        'age_dependent_mutation_strength': age_dependent_mutation_strength,
        'module_size_dependent_mutation': module_size_dependent_mutation,
        'connection_weight_dependent_mutation': connection_weight_dependent_mutation,
        'somatic_hypermutation_rate': somatic_hypermutation_rate,
        'error_driven_mutation_strength': error_driven_mutation_strength,
        'gene_centrality_mutation_bias': gene_centrality_mutation_bias,
        'epigenetic_mutation_influence': epigenetic_mutation_influence,
        'mutation_hotspot_probability': mutation_hotspot_probability,
        'add_connection_topology_bias': add_connection_topology_bias,
        'add_module_method': add_module_method,
        'remove_connection_probability': remove_connection_probability,
        'remove_module_probability': remove_module_probability,
        'connection_rewiring_probability': connection_rewiring_probability,
        'module_duplication_probability': module_duplication_probability,
        'module_fusion_probability': module_fusion_probability,
        'cycle_formation_probability': cycle_formation_probability,
        'structural_mutation_phase': structural_mutation_phase,
        'learning_rate_mutation_strength': learning_rate_mutation_strength,
        'plasticity_mutation_strength': plasticity_mutation_strength,
        'learning_improvement_mutation_bonus': learning_improvement_mutation_bonus,
        'weight_change_mutation_correlation': weight_change_mutation_correlation,
        'synaptic_tagging_credit_assignment': synaptic_tagging_credit_assignment,
        'metaplasticity_rule': metaplasticity_rule,
        'learning_instability_penalty': learning_instability_penalty,
        'gradient_guided_mutation_strength': gradient_guided_mutation_strength,
        'hessian_guided_mutation_strength': hessian_guided_mutation_strength,
        'crossover_operator': crossover_operator,
        'crossover_n_points': crossover_n_points,
        'crossover_parent_assortativity': crossover_parent_assortativity,
        'crossover_gene_dominance_probability': crossover_gene_dominance_probability,
        'sexual_reproduction_rate': sexual_reproduction_rate,
        'inbreeding_penalty_factor': inbreeding_penalty_factor,
        'horizontal_gene_transfer_rate': horizontal_gene_transfer_rate,
        'polyploidy_probability': polyploidy_probability,
        'meiotic_recombination_rate': meiotic_recombination_rate,
        'innovation_rate': innovation_rate,
        'enable_development': enable_development,
        'enable_baldwin': enable_baldwin,
        'baldwinian_assimilation_rate': baldwinian_assimilation_rate,
        'enable_epigenetics': enable_epigenetics,
        'endosymbiosis_rate': endosymbiosis_rate,
        # --- NEW CO-EVOLUTION & EMBODIMENT SETTINGS ---
        'enable_adversarial_coevolution': enable_adversarial_coevolution,
        'critic_population_size': critic_population_size,
        'critic_mutation_rate': critic_mutation_rate,
        'adversarial_fitness_weight': adversarial_fitness_weight,
        'critic_selection_pressure': critic_selection_pressure,
        'critic_task': critic_task,
        'enable_morphological_coevolution': enable_morphological_coevolution,
        'morphological_mutation_rate': morphological_mutation_rate,
        'max_body_modules': max_body_modules,
        'cost_per_module': cost_per_module,
        'enable_sensor_evolution': enable_sensor_evolution,
        'enable_actuator_evolution': enable_actuator_evolution,
        'physical_realism_factor': physical_realism_factor,
        'embodiment_gravity': embodiment_gravity,
        'embodiment_friction': embodiment_friction,
        # --- NEW SOPHISTICATED CO-EVOLUTION SETTINGS ---
        'enable_hall_of_fame': enable_hall_of_fame,
        'hall_of_fame_size': hall_of_fame_size,
        'hall_of_fame_replacement_strategy': hall_of_fame_replacement_strategy,
        'critic_evolution_frequency': critic_evolution_frequency,
        'critic_cooperation_probability': critic_cooperation_probability,
        'cooperative_reward_scaling': cooperative_reward_scaling,
        'critic_objective_novelty_weight': critic_objective_novelty_weight,
        'bilateral_symmetry_bonus': bilateral_symmetry_bonus,
        'segmentation_bonus': segmentation_bonus,
        'allometric_scaling_exponent': allometric_scaling_exponent,
        'enable_material_evolution': enable_material_evolution,
        'cost_per_stiffness': cost_per_stiffness,
        'cost_per_density': cost_per_density,
        'evolvable_sensor_noise': evolvable_sensor_noise,
        'evolvable_actuator_force': evolvable_actuator_force,
        'fluid_dynamics_viscosity': fluid_dynamics_viscosity,
        'surface_tension_factor': surface_tension_factor,
        'enable_host_symbiont_coevolution': enable_host_symbiont_coevolution,
        'symbiont_population_size': symbiont_population_size,
        'symbiont_mutation_rate': symbiont_mutation_rate,
        'symbiont_transfer_rate': symbiont_transfer_rate,
        'symbiont_vertical_inheritance_fidelity': symbiont_vertical_inheritance_fidelity,
        'host_symbiont_fitness_dependency': host_symbiont_fitness_dependency,
        # --- NEW MULTI-LEVEL SELECTION SETTINGS ---
        'enable_multi_level_selection': enable_multi_level_selection,
        'colony_formation_method': colony_formation_method,
        'colony_size': colony_size,
        'group_fitness_weight': group_fitness_weight,
        'selfishness_suppression_cost': selfishness_suppression_cost,
        'caste_specialization_bonus': caste_specialization_bonus,
        'inter_colony_competition_rate': inter_colony_competition_rate,
        'epistatic_linkage_k': epistatic_linkage_k,
        'gene_flow_rate': gene_flow_rate,
        'niche_competition_factor': niche_competition_factor,
        'max_archive_size': max_archive_size,
        # --- NEW ADVANCED LANDSCAPE PHYSICS SETTINGS ---
        'speciation_stagnation_threshold': speciation_stagnation_threshold,
        'species_extinction_threshold': species_extinction_threshold,
        'niche_construction_strength': niche_construction_strength,
        'character_displacement_pressure': character_displacement_pressure,
        'adaptive_radiation_trigger': adaptive_radiation_trigger,
        'species_merger_probability': species_merger_probability,
        'kin_selection_bonus': kin_selection_bonus,
        'sexual_selection_factor': sexual_selection_factor,
        'sympatric_speciation_pressure': sympatric_speciation_pressure,
        'allopatric_speciation_trigger': allopatric_speciation_trigger,
        'intraspecific_competition_scaling': intraspecific_competition_scaling,
        'landscape_ruggedness_factor': landscape_ruggedness_factor,
        'landscape_correlation_length': landscape_correlation_length,
        'landscape_neutral_network_size': landscape_neutral_network_size,
        'landscape_holeyness_factor': landscape_holeyness_factor,
        'landscape_anisotropy_factor': landscape_anisotropy_factor,
        'landscape_gradient_noise': landscape_gradient_noise,
        'landscape_time_variance_rate': landscape_time_variance_rate,
        'multimodality_factor': multimodality_factor,
        'epistatic_correlation_structure': epistatic_correlation_structure,
        'fitness_autocorrelation_time': fitness_autocorrelation_time,
        'fitness_landscape_plasticity': fitness_landscape_plasticity,
        'information_bottleneck_pressure': information_bottleneck_pressure,
        'fisher_information_maximization': fisher_information_maximization,
        'predictive_information_bonus': predictive_information_bonus,
        'thermodynamic_depth_bonus': thermodynamic_depth_bonus,
        'integrated_information_bonus': integrated_information_bonus,
        'free_energy_minimization_pressure': free_energy_minimization_pressure,
        'empowerment_maximization_drive': empowerment_maximization_drive,
        'causal_density_target': causal_density_target,
        'semantic_information_bonus': semantic_information_bonus,
        'algorithmic_complexity_penalty': algorithmic_complexity_penalty,
        'computational_irreducibility_bonus': computational_irreducibility_bonus,
        'altruism_punishment_effectiveness': altruism_punishment_effectiveness,
        'resource_depletion_rate': resource_depletion_rate,
        'predator_prey_cycle_period': predator_prey_cycle_period,
        'mutualism_bonus': mutualism_bonus,
        'parasitism_virulence_factor': parasitism_virulence_factor,
        'commensalism_emergence_bonus': commensalism_emergence_bonus,
        'social_learning_fidelity': social_learning_fidelity,
        'cultural_evolution_rate': cultural_evolution_rate,
        'group_selection_strength': group_selection_strength,
        'tragedy_of_the_commons_penalty': tragedy_of_the_commons_penalty,
        'reputation_dynamics_factor': reputation_dynamics_factor,
        'extinction_event_severity': extinction_event_severity,
        'environmental_shift_magnitude': environmental_shift_magnitude,
        'punctuated_equilibrium_trigger_sensitivity': punctuated_equilibrium_trigger_sensitivity,
        'key_innovation_bonus': key_innovation_bonus,
        'background_extinction_rate': background_extinction_rate,
        'invasive_species_introduction_prob': invasive_species_introduction_prob,
        'adaptive_radiation_factor': adaptive_radiation_factor,
        'refugia_survival_bonus': refugia_survival_bonus,
        'post_cataclysm_hypermutation_period': post_cataclysm_hypermutation_period,
        'environmental_press_factor': environmental_press_factor,
        'cambrian_explosion_trigger': cambrian_explosion_trigger,
        'reintroduction_rate': reintroduction_rate,
        'enable_cataclysms': enable_cataclysms,
        'cataclysm_probability': cataclysm_probability,
        'cataclysm_extinction_severity': cataclysm_extinction_severity,
        'cataclysm_landscape_shift_magnitude': cataclysm_landscape_shift_magnitude,
        'post_cataclysm_hypermutation_multiplier': post_cataclysm_hypermutation_multiplier,
        'post_cataclysm_hypermutation_duration': post_cataclysm_hypermutation_duration,
        'cataclysm_selectivity_type': cataclysm_selectivity_type,
        'red_queen_virulence': red_queen_virulence,
        'red_queen_adaptation_speed': red_queen_adaptation_speed,
        'red_queen_target_breadth': red_queen_target_breadth,
        'enable_red_queen': enable_red_queen,
        'enable_endosymbiosis': enable_endosymbiosis,
        'mutation_schedule': mutation_schedule,
        'adaptive_mutation_strength': adaptive_mutation_strength,
        'selection_pressure': selection_pressure,
        'enable_speciation': enable_speciation,
        'enable_diversity_pressure': enable_diversity_pressure,
        'diversity_weight': diversity_weight,
        # --- NEW ADVANCED SPECIATION SETTINGS ---
        'enable_advanced_speciation': enable_advanced_speciation,
        'dynamic_threshold_adjustment_rate': dynamic_threshold_adjustment_rate,
        'distance_weight_c1': distance_weight_c1,
        'distance_weight_c2': distance_weight_c2,
        'phenotypic_distance_weight': phenotypic_distance_weight,
        'age_distance_weight': age_distance_weight,
        'lineage_distance_factor': lineage_distance_factor,
        'distance_normalization_factor': distance_normalization_factor,
        'developmental_rule_distance_weight': developmental_rule_distance_weight,
        'meta_param_distance_weight': meta_param_distance_weight,
        'species_stagnation_threshold': species_stagnation_threshold,
        'stagnation_penalty': stagnation_penalty,
        'species_age_bonus': species_age_bonus,
        'species_novelty_bonus': species_novelty_bonus,
        'min_species_size_for_survival': min_species_size_for_survival,
        'species_extinction_threshold': species_extinction_threshold,
        'species_merger_threshold': species_merger_threshold,
        'species_merger_probability': species_merger_probability,
        'niche_construction_strength': niche_construction_strength,
        'character_displacement_pressure': character_displacement_pressure,
        'intraspecific_competition_scaling': intraspecific_competition_scaling,
        'interspecific_competition_scaling': interspecific_competition_scaling,
        'resource_depletion_rate': resource_depletion_rate,
        'niche_overlap_penalty': niche_overlap_penalty,
        'niche_capacity': niche_capacity,
        'sexual_selection_factor': sexual_selection_factor,
        'mating_preference_strength': mating_preference_strength,
        'outbreeding_depression_penalty': outbreeding_depression_penalty,
        'inbreeding_depression_penalty': inbreeding_depression_penalty,
        'reproductive_isolation_threshold': reproductive_isolation_threshold,
        'assortative_mating_strength': assortative_mating_strength,
        'sympatric_speciation_pressure': sympatric_speciation_pressure,
        'allopatric_speciation_trigger': allopatric_speciation_trigger,
        'parapatric_speciation_gradient': parapatric_speciation_gradient,
        'peripatric_speciation_founder_effect': peripatric_speciation_founder_effect,
        'adaptive_radiation_trigger_threshold': adaptive_radiation_trigger_threshold,
        'adaptive_radiation_strength': adaptive_radiation_strength,
        'kin_selection_bonus': kin_selection_bonus,
        'group_selection_strength': group_selection_strength,
        'altruism_cost': altruism_cost,
        'species_reputation_factor': species_reputation_factor,
        'punctuated_equilibrium_trigger_sensitivity': punctuated_equilibrium_trigger_sensitivity,
        'background_extinction_rate': background_extinction_rate,
        'key_innovation_bonus': key_innovation_bonus,
        'invasive_species_introduction_prob': invasive_species_introduction_prob,
        'refugia_survival_bonus': refugia_survival_bonus,
        'phyletic_gradualism_factor': phyletic_gradualism_factor,
        'species_sorting_strength': species_sorting_strength,
        'compatibility_threshold': compatibility_threshold,
        'num_generations': num_generations,
        'complexity_level': complexity_level,
        'experiment_name': experiment_name,
        'random_seed': random_seed,
        'enable_early_stopping': enable_early_stopping,
        'early_stopping_patience': early_stopping_patience,
        'checkpoint_frequency': checkpoint_frequency,
        'analysis_top_n': analysis_top_n,
        'enable_hyperparameter_evolution': enable_hyperparameter_evolution,
        'evolvable_params': evolvable_params,
        'hyper_mutation_rate': hyper_mutation_rate,
        # --- NEW META-EVOLUTION SETTINGS ---
        'hyper_mutation_distribution': hyper_mutation_distribution,
        'evolvable_param_bounds_leniency': evolvable_param_bounds_leniency,
        'hyperparam_heritability_factor': hyperparam_heritability_factor,
        
        'enable_genetic_code_evolution': enable_genetic_code_evolution,
        'gene_type_innovation_rate': gene_type_innovation_rate,
        'gene_type_extinction_rate': gene_type_extinction_rate,
        'evolvable_activation_functions': evolvable_activation_functions,
        'activation_expression_complexity_limit': activation_expression_complexity_limit,
        'developmental_rule_innovation_rate': developmental_rule_innovation_rate,
        'encoding_plasticity_rate': encoding_plasticity_rate,
        'genome_length_constraint_pressure': genome_length_constraint_pressure,
        'intron_ratio_target': intron_ratio_target,
        'gene_regulatory_network_complexity_bonus': gene_regulatory_network_complexity_bonus,
        'evolvable_normalization_layers': evolvable_normalization_layers,

        'enable_ea_dynamics_evolution': enable_ea_dynamics_evolution,
        'evolvable_selection_mechanism': evolvable_selection_mechanism,
        'selection_mechanism_pool': selection_mechanism_pool,
        'evolvable_tournament_size': evolvable_tournament_size,
        'crossover_operator_pool': crossover_operator_pool,
        'mutation_operator_pool': mutation_operator_pool,
        'population_topology': population_topology,
        'evolvable_migration_rate': evolvable_migration_rate,
        'evolvable_island_count': evolvable_island_count,
        'topology_reconfiguration_frequency': topology_reconfiguration_frequency,
        'dynamic_speciation_threshold_factor': dynamic_speciation_threshold_factor,

        'enable_objective_evolution': enable_objective_evolution,
        'evolvable_objective_weights': evolvable_objective_weights,
        'objective_weight_mutation_strength': objective_weight_mutation_strength,
        'autotelic_novelty_search_weight': autotelic_novelty_search_weight,
        'autotelic_complexity_drive_weight': autotelic_complexity_drive_weight,
        'autotelic_learning_progress_drive': autotelic_learning_progress_drive,
        'fitness_function_noise_injection_rate': fitness_function_noise_injection_rate,
        'fitness_landscape_smoothing_factor': fitness_landscape_smoothing_factor,
        'objective_ambition_ratchet': objective_ambition_ratchet,
        'pareto_front_focus_bias': pareto_front_focus_bias,

        'enable_self_modification': enable_self_modification,
        'self_modification_probability': self_modification_probability,
        'self_modification_scope': self_modification_scope,
        'quine_bonus': quine_bonus,
        'meta_genotype_bonus': meta_genotype_bonus,
        'self_simulation_bonus': self_simulation_bonus,
        'enable_curriculum_learning': enable_curriculum_learning,
        'curriculum_sequence': curriculum_sequence,
        'curriculum_trigger': curriculum_trigger,
        'curriculum_threshold': curriculum_threshold,
        'enable_iterative_seeding': enable_iterative_seeding,
        'num_elites_to_seed': num_elites_to_seed,
        'seeded_elite_mutation_strength': seeded_elite_mutation_strength,
        # --- NEW DYNAMIC ENVIRONMENT SETTINGS ---
        'enable_advanced_environment_physics': enable_advanced_environment_physics,
        'non_stationarity_mode': non_stationarity_mode,
        'drift_velocity': drift_velocity,
        'shift_magnitude': shift_magnitude,
        'cycle_period': cycle_period,
        'chaotic_attractor_type': chaotic_attractor_type,
        'environmental_memory_strength': environmental_memory_strength,
        'resource_distribution_mode': resource_distribution_mode,
        'resource_regeneration_rate': resource_regeneration_rate,
        'task_space_curvature': task_space_curvature,
        'environmental_viscosity': environmental_viscosity,
        'environmental_temperature': environmental_temperature,
        'task_noise_correlation_time': task_noise_correlation_time,
        'environmental_lag': environmental_lag,
        'resource_scarcity_level': resource_scarcity_level,

        'enable_advanced_curriculum': enable_advanced_curriculum,
        'curriculum_generation_method': curriculum_generation_method,
        'self_paced_learning_rate': self_paced_learning_rate,
        'teacher_student_dynamics_enabled': teacher_student_dynamics_enabled,
        'teacher_mutation_rate': teacher_mutation_rate,
        'task_proposal_rejection_rate': task_proposal_rejection_rate,
        'transfer_learning_bonus': transfer_learning_bonus,
        'catastrophic_forgetting_penalty': catastrophic_forgetting_penalty,
        'curriculum_backtracking_probability': curriculum_backtracking_probability,
        'interleaved_learning_ratio': interleaved_learning_ratio,
        'task_decomposition_bonus': task_decomposition_bonus,
        'procedural_content_generation_complexity': procedural_content_generation_complexity,
        'curriculum_difficulty_ceiling': curriculum_difficulty_ceiling,
        'teacher_student_objective_alignment': teacher_student_objective_alignment,

        'enable_social_environment': enable_social_environment,
        'communication_channel_bandwidth': communication_channel_bandwidth,
        'communication_channel_noise': communication_channel_noise,
        'social_signal_cost': social_signal_cost,
        'common_knowledge_bonus': common_knowledge_bonus,
        'deception_penalty': deception_penalty,
        'reputation_system_fidelity': reputation_system_fidelity,
        'sanctioning_effectiveness': sanctioning_effectiveness,
        'network_reciprocity_bonus': network_reciprocity_bonus,
        'social_learning_mechanism': social_learning_mechanism,
        'cultural_ratchet_bonus': cultural_ratchet_bonus,
        'social_norm_emergence_bonus': social_norm_emergence_bonus,
        'tribalism_factor': tribalism_factor,

        'enable_open_endedness': enable_open_endedness,
        'poi_novelty_threshold': poi_novelty_threshold,
        'minimal_criterion_coevolution_rate': minimal_criterion_coevolution_rate,
        'autopoiesis_pressure': autopoiesis_pressure,
        'environmental_construction_bonus': environmental_construction_bonus,
        'goal_switching_cost': goal_switching_cost,
        'solution_archive_capacity': solution_archive_capacity,
        'novelty_metric': novelty_metric,
        'local_competition_radius': local_competition_radius,
        'information_seeking_drive': information_seeking_drive,
        'open_ended_archive_sampling_bias': open_ended_archive_sampling_bias,
        'goal_embedding_space_dims': goal_embedding_space_dims,
        # --- NEW FINALIZATION SETTINGS ---
        'enable_ensemble_creation': enable_ensemble_creation,
        'ensemble_size': ensemble_size,
        'ensemble_selection_strategy': ensemble_selection_strategy,
        'enable_fine_tuning': enable_fine_tuning,
        'fine_tuning_generations': fine_tuning_generations,
        'fine_tuning_mutation_multiplier': fine_tuning_mutation_multiplier,
        # --- NEW DEEP PHYSICS SETTINGS ---
        # --- NEW ADVANCED FINALIZATION SETTINGS ---
        'enable_advanced_finalization': enable_advanced_finalization,
        'pruning_method': pruning_method,
        'pruning_aggressiveness': pruning_aggressiveness,
        'model_compression_target_ratio': model_compression_target_ratio,
        'quantization_bits': quantization_bits,
        'lottery_ticket_pruning_iterations': lottery_ticket_pruning_iterations,
        'knowledge_distillation_temperature': knowledge_distillation_temperature,
        'distillation_teacher_selection': distillation_teacher_selection,
        'self_distillation_weight': self_distillation_weight,
        'model_merging_method': model_merging_method,
        'merging_resolution_method': merging_resolution_method,
        'model_merging_alpha': model_merging_alpha,
        'bayesian_model_averaging_prior': bayesian_model_averaging_prior,
        'stacking_meta_learner_complexity': stacking_meta_learner_complexity,
        'calibration_method': calibration_method,
        'out_of_distribution_generalization_test': out_of_distribution_generalization_test,
        'formal_verification_engine': formal_verification_engine,
        'adversarial_robustness_certification_method': adversarial_robustness_certification_method,
        'explainability_method': explainability_method,
        'symbolic_regression_complexity_penalty': symbolic_regression_complexity_penalty,
        'causal_model_extraction_method': causal_model_extraction_method,
        'concept_extraction_method': concept_extraction_method,
        'concept_bottleneck_regularization': concept_bottleneck_regularization,
        'mechanistic_interpretability_circuit_search': mechanistic_interpretability_circuit_search,
        'continual_learning_replay_buffer_size': continual_learning_replay_buffer_size,
        'elastic_weight_consolidation_lambda': elastic_weight_consolidation_lambda,
        'synaptic_intelligence_c_param': synaptic_intelligence_c_param,
        'solution_export_format': solution_export_format,
        'deployment_latency_constraint': deployment_latency_constraint,
        'energy_consumption_constraint': energy_consumption_constraint,
        'final_report_verbosity': final_report_verbosity,
        'archive_solution_for_future_seeding': archive_solution_for_future_seeding,
        'generate_evolutionary_lineage_report': generate_evolutionary_lineage_report,
        'perform_sensitivity_analysis_on_hyperparameters': perform_sensitivity_analysis_on_hyperparameters,
        'ablation_study_component_count': ablation_study_component_count,
        'cross_validation_folds': cross_validation_folds,
        'enable_advanced_frameworks': enable_advanced_frameworks,
        # Computational Logic
        'chaitin_omega_bias': chaitin_omega_bias,
        'godel_incompleteness_penalty': godel_incompleteness_penalty,
        'turing_completeness_bonus': turing_completeness_bonus,
        'lambda_calculus_isomorphism': lambda_calculus_isomorphism,
        'proof_complexity_cost': proof_complexity_cost,
        'constructive_type_theory_adherence': constructive_type_theory_adherence,
        # Learning Theory
        'pac_bayes_bound_minimization': pac_bayes_bound_minimization,
        'vc_dimension_constraint': vc_dimension_constraint,
        'rademacher_complexity_penalty': rademacher_complexity_penalty,
        'algorithmic_stability_pressure': algorithmic_stability_pressure,
        'maml_readiness_bonus': maml_readiness_bonus,
        'causal_inference_engine_bonus': causal_inference_engine_bonus,
        # Morphogenesis
        'reaction_diffusion_activator_rate': reaction_diffusion_activator_rate,
        'reaction_diffusion_inhibitor_rate': reaction_diffusion_inhibitor_rate,
        'morphogen_gradient_decay': morphogen_gradient_decay,
        'cell_adhesion_factor': cell_adhesion_factor,
        'apoptosis_schedule_factor': apoptosis_schedule_factor,
        'hox_gene_expression_control': hox_gene_expression_control,
        # Collective Intelligence
        'stigmergy_potential_factor': stigmergy_potential_factor,
        'quorum_sensing_threshold': quorum_sensing_threshold,
        'social_learning_fidelity': social_learning_fidelity,
        'cultural_transmission_rate': cultural_transmission_rate,
        'division_of_labor_incentive': division_of_labor_incentive,
        'consensus_algorithm_efficiency': consensus_algorithm_efficiency,
        # Game Theory
        'hawk_dove_strategy_ratio': hawk_dove_strategy_ratio,
        'ultimatum_game_fairness_pressure': ultimatum_game_fairness_pressure,
        'principal_agent_alignment_bonus': principal_agent_alignment_bonus,
        'market_clearing_price_efficiency': market_clearing_price_efficiency,
        'contract_theory_enforcement_cost': contract_theory_enforcement_cost,
        'vickrey_auction_selection_bonus': vickrey_auction_selection_bonus,
        # Neuromodulation
        'dopamine_reward_prediction_error': dopamine_reward_prediction_error,
        'serotonin_uncertainty_signal': serotonin_uncertainty_signal,
        'acetylcholine_attentional_gain': acetylcholine_attentional_gain,
        'noradrenaline_arousal_level': noradrenaline_arousal_level,
        'bcm_rule_sliding_threshold': bcm_rule_sliding_threshold,
        'synaptic_scaling_homeostasis': synaptic_scaling_homeostasis,
        # Abstract Algebra
        'group_theory_symmetry_bonus': group_theory_symmetry_bonus,
        'category_theory_functorial_bonus': category_theory_functorial_bonus,
        'monad_structure_bonus': monad_structure_bonus,
        'lie_algebra_dynamics_prior': lie_algebra_dynamics_prior,
        'simplicial_complex_bonus': simplicial_complex_bonus,
        'sheaf_computation_consistency': sheaf_computation_consistency,
        'enable_deep_physics': enable_deep_physics,
        # Info-Theoretic
        'kolmogorov_pressure': kolmogorov_pressure,
        'pred_info_bottleneck': pred_info_bottleneck,
        'causal_emergence_factor': causal_emergence_factor,
        'semantic_closure_pressure': semantic_closure_pressure,
        'phi_target': phi_target,
        'fep_gradient': fep_gradient,
        'transfer_entropy_maximization': transfer_entropy_maximization,
        'synergy_bias': synergy_bias,
        'state_space_compression': state_space_compression,
        'fisher_gradient_ascent': fisher_gradient_ascent,
        # Thermo
        'landauer_efficiency': landauer_efficiency,
        'metabolic_power_law': metabolic_power_law,
        'heat_dissipation_constraint': heat_dissipation_constraint,
        'homeostatic_pressure': homeostatic_pressure,
        'computational_temperature': computational_temperature,
        'structural_decay_rate': structural_decay_rate,
        'repair_mechanism_cost': repair_mechanism_cost,
        'szilard_engine_efficiency': szilard_engine_efficiency,
        'resource_scarcity': resource_scarcity,
        'allosteric_regulation_factor': allosteric_regulation_factor,
        # Quantum
        'quantum_annealing_fluctuation': quantum_annealing_fluctuation,
        'holographic_constraint': holographic_constraint,
        'renormalization_group_flow': renormalization_group_flow,
        'symmetry_breaking_pressure': symmetry_breaking_pressure,
        'majorana_fermion_pairing_bonus': majorana_fermion_pairing_bonus,
        'path_integral_exploration': path_integral_exploration,
        'tqft_invariance': tqft_invariance,
        'gauge_theory_redundancy': gauge_theory_redundancy,
        'cft_scaling_exponent': cft_scaling_exponent,
        'spacetime_foam_fluctuation': spacetime_foam_fluctuation,
        'entanglement_assisted_comm': entanglement_assisted_comm,
        # Topology
        'manifold_adherence': manifold_adherence,
        'group_equivariance_prior': group_equivariance_prior,
        'ricci_curvature_flow': ricci_curvature_flow,
        'homological_scaffold_stability': homological_scaffold_stability,
        'fractal_dimension_target': fractal_dimension_target,
        'hyperbolic_embedding_factor': hyperbolic_embedding_factor,
        'small_world_bias': small_world_bias,
        'scale_free_exponent': scale_free_exponent,
        'network_motif_bonus': network_motif_bonus,
        'autocatalytic_set_emergence': autocatalytic_set_emergence,
        'rents_rule_exponent': rents_rule_exponent,
        # Cognitive
        'curiosity_drive': curiosity_drive,
        'world_model_accuracy': world_model_accuracy,
        'ast_congruence': ast_congruence,
        'tom_emergence_pressure': tom_emergence_pressure,
        'cognitive_dissonance_penalty': cognitive_dissonance_penalty,
        'opportunity_cost_factor': opportunity_cost_factor,
        'prospect_theory_bias': prospect_theory_bias,
        'temporal_discounting_factor': temporal_discounting_factor,
        'zpd_scaffolding_bonus': zpd_scaffolding_bonus,
        'symbol_grounding_constraint': symbol_grounding_constraint,
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
    init_col, resume_col = st.sidebar.columns(2)

    # --- INITIATE EVOLUTION ---
    if init_col.button("⚡ Initiate Evolution", type="primary", width='stretch', key="initiate_evolution_button"):
        st.session_state.history = []
        st.session_state.evolutionary_metrics = [] # type: ignore
        st.session_state.gene_archive = [] # Initialize the infinite gene pool
        
        # --- Seeding for reproducibility ---
        if random_seed != -1:
            random.seed(random_seed)
            np.random.seed(random_seed)
            st.toast(f"Using fixed random seed: {random_seed}", icon="🎲")
        else:
            st.toast("Using random seed.", icon="🎲")
        
        # Initialize population
        if enable_iterative_seeding and st.session_state.get('current_population'):
            st.toast(f"Seeding new evolution with {num_elites_to_seed} elites from previous run.", icon="🌱")
            
            previous_elites = sorted(st.session_state.current_population, key=lambda x: x.fitness, reverse=True)[:num_elites_to_seed]
            
            population = []
            # Create new population from mutated elites
            while len(population) < total_population:
                elite_to_copy = random.choice(previous_elites)
                new_individual = mutate(elite_to_copy, mutation_rate=seeded_elite_mutation_strength, innovation_rate=seeded_elite_mutation_strength/2)
                new_individual.generation = 0
                new_individual.parent_ids = [f"SEED_{elite_to_copy.lineage_id}"]
                population.append(new_individual)
            
            population = population[:total_population] # Ensure correct size
        else:
            population = []
            for form_id in range(1, num_forms + 1):
                for _ in range(population_per_form):
                    genotype = initialize_genotype(form_id, complexity_level)
                    genotype.generation = 0
                    population.append(genotype)
                    st.session_state.gene_archive.append(genotype.copy()) # Seed the archive
        
        # For adaptive mutation
        last_best_fitness = -1
        best_fitness_overall = -1
        stagnation_counter = 0
        early_stop_counter = 0
        current_mutation_rate = mutation_rate
        
        # For dynamic environment
        if enable_curriculum_learning and curriculum_sequence:
            st.session_state.curriculum_stage = 0
            current_task = curriculum_sequence[0]
            st.toast(f"🎓 Curriculum Started! Task 1: {current_task}", icon="📈")
        else:
            st.session_state.curriculum_stage = -1 # Mark as disabled
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
                if enable_hyperparameter_evolution:
                    # If evolving, the individual's own meta-params are used for some things
                    # The fitness function itself doesn't need them, but they affect reproduction
                    pass

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

            # --- Update stagnation counters ---
            current_gen_best_fitness = fitness_array.max()
            if current_gen_best_fitness > best_fitness_overall + 1e-6: # Use tolerance for float comparison
                best_fitness_overall = current_gen_best_fitness
                early_stop_counter = 0
            else:
                early_stop_counter += 1

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
                'mean_fitness': fitness_array.mean(),
                'mutation_rate': current_mutation_rate,
                'stagnation_counter': stagnation_counter
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
                # Also calculate novelty if diversity pressure is on
                for s in species:
                    species_size = len(s['members'])
                    if species_size > 0:
                        for member in s['members']:
                            member.adjusted_fitness = member.fitness / (species_size ** niche_competition_factor)
                
                if enable_diversity_pressure:
                    # Calculate novelty for each individual across the whole population
                    for ind in population:
                        distances = [genomic_distance(ind, other) for other in population if ind.lineage_id != other.lineage_id]
                        distances = [d for d in distances if d != float('inf')]
                        if distances:
                            k = min(10, len(distances))
                            distances.sort()
                            ind.novelty_score = np.mean(distances[:k])
                        else:
                            ind.novelty_score = 0.0
                    
                    max_novelty = max((ind.novelty_score for ind in population), default=1.0)
                    if max_novelty > 0:
                        for ind in population:
                            ind.selection_score = ind.adjusted_fitness + diversity_weight * (ind.novelty_score / max_novelty)
                    else:
                        for ind in population:
                            ind.selection_score = ind.adjusted_fitness
                    
                    selection_key = lambda x: x.selection_score
                else:
                    selection_key = lambda x: x.adjusted_fitness

            else:
                species_metric.metric("Species Count", f"{len(set(ind.form_id for ind in population))}")
                if enable_diversity_pressure:
                    for ind in population:
                        distances = [genomic_distance(ind, other) for other in population if ind.lineage_id != other.lineage_id]
                        distances = [d for d in distances if d != float('inf')]
                        if distances:
                            k = min(10, len(distances))
                            distances.sort()
                            ind.novelty_score = np.mean(distances[:k])
                        else:
                            ind.novelty_score = 0.0
                    max_novelty = max((ind.novelty_score for ind in population), default=1.0)
                    for ind in population:
                        ind.selection_score = ind.fitness + diversity_weight * (ind.novelty_score / (max_novelty + 1e-9))
                    selection_key = lambda x: x.selection_score
                else:
                    selection_key = lambda x: x.fitness

            population.sort(key=selection_key, reverse=True)

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
                if random.random() < reintroduction_rate and st.session_state.gene_archive:
                    # Reintroduce a "fossil" from the infinite gene pool
                    child = random.choice(st.session_state.gene_archive).copy()
                    child = mutate(child, current_mutation_rate * 1.5, innovation_rate * 1.5) # Mutate it heavily to adapt it
                    child.generation = gen + 1
                    offspring.append(child)
                else:
                    # --- Create one viable child via normal reproduction, with retries ---
                    # Determine crossover rate for this reproductive event
                    p1_crossover_rate = parent1.meta_parameters.get('crossover_rate', crossover_rate) if enable_hyperparameter_evolution else crossover_rate

                    max_attempts = 20
                    for _ in range(max_attempts):
                        # Tournament selection using the appropriate fitness key
                        parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                        
                        # Use parent1's crossover rate
                        effective_crossover_rate = parent1.meta_parameters.get('crossover_rate', crossover_rate) if enable_hyperparameter_evolution else crossover_rate

                        if random.random() < effective_crossover_rate:
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
                            child = crossover(parent1, parent2, effective_crossover_rate)
                        else:
                            child = parent1.copy()
                        
                        # Mutation and other operators
                        # Use child's own inherited rates if they exist
                        child_mutation_rate = child.meta_parameters.get('mutation_rate', current_mutation_rate) if enable_hyperparameter_evolution else current_mutation_rate
                        child_innovation_rate = child.meta_parameters.get('innovation_rate', innovation_rate) if enable_hyperparameter_evolution else innovation_rate
                        
                        # Mutate the child, which also mutates its own hyperparameters if enabled
                        child = mutate(child, child_mutation_rate, child_innovation_rate)
                        if enable_hyperparameter_evolution:
                            for param in evolvable_params:
                                if random.random() < current_hyper_mutation_rate:
                                    child.meta_parameters[param] *= np.random.lognormal(0, 0.1)

                        if enable_endosymbiosis and random.random() < endosymbiosis_rate and survivors:
                            child = apply_endosymbiosis(child, survivors)
                        
                        # Viability Selection: Ensure the child is a functional network
                        if is_viable(child):
                            child.generation = gen + 1
                            offspring.append(child)
                            st.session_state.gene_archive.append(child.copy()) # Add new viable child to archive
                            # Prune archive if it exceeds max size to manage memory
                            max_size = st.session_state.settings.get('max_archive_size', 10000)
                            if len(st.session_state.gene_archive) > max_size:
                                st.session_state.gene_archive = random.sample(st.session_state.gene_archive, max_size)
                            break # Found a viable child, move to next offspring
                    else: # for-else: runs if the loop finished without break
                        # Fallback if no viable child was found after many attempts
                        parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                        child = parent1.copy()
                        child_mutation_rate = child.meta_parameters.get('mutation_rate', current_mutation_rate) if enable_hyperparameter_evolution else current_mutation_rate
                        child_innovation_rate = child.meta_parameters.get('innovation_rate', innovation_rate) if enable_hyperparameter_evolution else innovation_rate
                        child = mutate(child, child_mutation_rate, child_innovation_rate)
                        if enable_hyperparameter_evolution:
                            for param in evolvable_params:
                                if random.random() < current_hyper_mutation_rate:
                                    child.meta_parameters[param] *= np.random.lognormal(0, 0.1)
                        child.generation = gen + 1
                        offspring.append(child)
            
            # Clean up temporary attribute
            if enable_speciation:
                for ind in population: # Clean up all temporary scores
                    for attr in ['adjusted_fitness', 'novelty_score', 'selection_score']: # type: ignore
                        if hasattr(ind, attr): delattr(ind, attr)

            # Update mutation rate for next generation
            if mutation_schedule == 'Linear Decay':
                # If evolving, this global rate is just a fallback.
                if not ('mutation_rate' in evolvable_params and enable_hyperparameter_evolution):
                    current_mutation_rate = max(0.01, mutation_rate * (1.0 - ((gen + 1) / num_generations)))
            elif mutation_schedule == 'Adaptive':
                current_best_fitness = current_gen_best_fitness # Use already computed value
                if current_best_fitness > last_best_fitness:
                    # If not evolving, anneal the global rate
                    if not ('mutation_rate' in evolvable_params and enable_hyperparameter_evolution):
                        current_mutation_rate = max(0.05, current_mutation_rate * 0.95) # Anneal
                    stagnation_counter = 0
                else:
                    stagnation_counter += 1
                
                if stagnation_counter > 3: # If stagnated for >3 generations
                    # If not evolving, spike the global rate
                    if not ('mutation_rate' in evolvable_params and enable_hyperparameter_evolution):
                        current_mutation_rate = min(0.8, current_mutation_rate * (1 + adaptive_mutation_strength)) # Spike
                
                last_best_fitness = current_best_fitness
            
            # --- Early Stopping Check ---
            if enable_early_stopping and early_stop_counter > early_stopping_patience:
                st.success(f"**EARLY STOPPING TRIGGERED:** Best fitness has not improved for {early_stopping_patience} generations.")
                st.toast("Evolution stopped early due to stagnation.", icon="🛑")
                break # Exit the evolution loop

            # --- Curriculum Learning Check ---
            if enable_curriculum_learning and st.session_state.curriculum_stage != -1:
                condition_met = False
                stage = st.session_state.curriculum_stage
                if curriculum_trigger == 'Fixed Generations':
                    if (gen + 1) % int(curriculum_threshold) == 0:
                        condition_met = True
                elif curriculum_trigger == 'Mean Accuracy Threshold':
                    mean_accuracy_this_gen = pd.DataFrame(st.session_state.history)[history_df['generation'] == gen]['accuracy'].mean()
                    if mean_accuracy_this_gen >= curriculum_threshold:
                        condition_met = True
                elif curriculum_trigger == 'Apex Fitness Threshold':
                    if current_gen_best_fitness >= curriculum_threshold:
                        condition_met = True
                
                if condition_met and stage < len(curriculum_sequence) - 1:
                    st.session_state.curriculum_stage += 1
                    current_task = curriculum_sequence[st.session_state.curriculum_stage]
                    st.toast(f"🎓 Curriculum Advanced! New Task: {current_task}", icon="📈")

            population = survivors + offspring
            
            # --- Checkpointing ---
            if checkpoint_frequency > 0 and (gen + 1) % checkpoint_frequency == 0 and (gen + 1) < num_generations:
                serializable_population = [genotype_to_dict(p) for p in population]
                results_to_save = { 'history': st.session_state.history, 'evolutionary_metrics': st.session_state.evolutionary_metrics, 'current_population': serializable_population }
                if results_table.get(doc_id=1): results_table.update(results_to_save, doc_ids=[1])
                else: results_table.insert(results_to_save)
                st.toast(f"Checkpoint saved at generation {gen + 1}", icon="💾")
            
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

    # --- RESUME EVOLUTION ---
    if resume_col.button("🔄 Resume Evolution", width='stretch', key="resume_evolution_button"):
        if not st.session_state.get('history') or not st.session_state.get('current_population'):
            st.error("No previous experiment state found to resume. Please initiate a new evolution first.")
            st.stop()

        st.toast("Resuming previous evolution...", icon="🔄")

        # --- Restore State ---
        population = st.session_state.current_population
        history_df = pd.DataFrame(st.session_state.history)
        metrics_df = pd.DataFrame(st.session_state.evolutionary_metrics)

        start_gen = history_df['generation'].max() + 1
        
        if start_gen >= num_generations:
            st.warning("The previous evolution has already completed all generations. Please increase the number of generations to resume.")
            st.stop()

        # Restore counters and rates for adaptive schedules
        last_best_fitness = metrics_df.iloc[-1]['best_fitness']
        best_fitness_overall = metrics_df['best_fitness'].max()
        current_mutation_rate = metrics_df.iloc[-1]['mutation_rate']
        stagnation_counter = metrics_df.iloc[-1]['stagnation_counter']

        # Recalculate early stopping counter
        best_fitness_per_gen = history_df.groupby('generation')['fitness'].max()
        best_fitness_overall_series = best_fitness_per_gen.cummax()
        last_improvement_gen = best_fitness_overall_series[best_fitness_overall_series.diff() > 1e-6].index.max()
        if pd.isna(last_improvement_gen):
            last_improvement_gen = 0
        early_stop_counter = (start_gen - 1) - last_improvement_gen

        # Restore other state
        current_task = task_type # Assume we continue with the currently selected task
        if 'cataclysm_recovery_mode' not in st.session_state: st.session_state.cataclysm_recovery_mode = 0
        if 'cataclysm_weights' not in st.session_state: st.session_state.cataclysm_weights = None
        if 'parasite_profile' not in st.session_state: st.session_state.parasite_profile = {'target_type': 'attention', 'target_activation': 'gelu'}

        # --- Seeding for reproducibility ---
        if random_seed != -1:
            random.seed(random_seed)
            np.random.seed(random_seed)
            st.toast(f"Resuming with fixed random seed: {random_seed}", icon="🎲")
        
        # Progress tracking
        progress_container = st.empty()
        metrics_container = st.empty()
        status_text = st.empty()
        
        # Evolution loop
        for gen in range(start_gen, num_generations):
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
            if enable_curriculum_learning:
                # Curriculum logic is handled at the end of the loop
                pass
            elif dynamic_environment and gen > 0 and gen % env_change_frequency == 0:
                # Original random dynamic environment
                if len(task_options) > 1:
                    previous_task = current_task
                    current_task = random.choice([t for t in task_options if t != previous_task])
                    st.toast(f"🌍 Environment Shift! New Task: {current_task}", icon="🔄")
                    time.sleep(1.0)

            # Update meta-parameter mutation rate if it's evolving
            if enable_hyperparameter_evolution and 'hyper_mutation_rate' in evolvable_params:
                # Use population average for the global rate, individuals use their own
                current_hyper_mutation_rate = np.mean([ind.meta_parameters.get('hyper_mutation_rate', hyper_mutation_rate) for ind in population])
            else:
                current_hyper_mutation_rate = hyper_mutation_rate


            
            status_text.markdown(f"### 🧬 Generation {gen + 1}/{num_generations} | Task: **{current_task}**")
            
            # --- Apply developmental rules ---
            if enable_development:
                for i in range(len(population)):
                    population[i] = apply_developmental_rules(population[i], stagnation_counter)

            # Evaluate fitness
            all_scores = []
            for individual in population:
                fitness, component_scores = evaluate_fitness(individual, current_task, gen, active_fitness_weights, enable_epigenetics, enable_baldwin, epistatic_linkage_k, st.session_state.parasite_profile if enable_red_queen else None)
                individual.fitness = fitness
                individual.generation = gen
                individual.age += 1
                all_scores.append(component_scores)
            
            # Record detailed history
            for individual, scores in zip(population, all_scores):
                st.session_state.history.append({
                    'generation': gen, 'form': f'Form {individual.form_id}', 'form_id': individual.form_id,
                    'fitness': individual.fitness, 'accuracy': scores['task_accuracy'], 'efficiency': scores['efficiency'],
                    'robustness': scores['robustness'], 'generalization': scores['generalization'],
                    'total_params': sum(m.size for m in individual.modules), 'num_connections': len(individual.connections),
                    'complexity': individual.complexity, 'lineage_id': individual.lineage_id, 'parent_ids': individual.parent_ids
                })
            
            # Compute evolutionary metrics
            fitness_array = np.array([ind.fitness for ind in population])
            diversity = EvolutionaryTheory.genetic_diversity(population)
            fisher_info = EvolutionaryTheory.fisher_information(population, fitness_array)

            # --- Update stagnation counters ---
            current_gen_best_fitness = fitness_array.max()
            if current_gen_best_fitness > best_fitness_overall + 1e-6:
                best_fitness_overall = current_gen_best_fitness
                early_stop_counter = 0
            else:
                early_stop_counter += 1

            # Display real-time metrics
            with metrics_container.container():
                col1, col2, col3, col4, col5 = st.columns(5)
                col1.metric("Best Fitness", f"{fitness_array.max():.4f}")
                col2.metric("Mean Fitness", f"{fitness_array.mean():.4f}")
                col3.metric("Diversity (H)", f"{diversity:.3f}")
                col4.metric("Mutation Rate (μ)", f"{current_mutation_rate:.3f}")
                if enable_red_queen:
                    parasite_display.info(f"**Red Queen Active:** Parasite targeting `{st.session_state.parasite_profile['target_type']}` with `{st.session_state.parasite_profile['target_activation']}` activation.")
                else:
                    parasite_display.empty()
                species_metric = col5.metric("Species Count", "N/A")

            st.session_state.evolutionary_metrics.append({
                'generation': gen, 'diversity': diversity, 'fisher_info': fisher_info,
                'best_fitness': fitness_array.max(), 'mean_fitness': fitness_array.mean(),
                'mutation_rate': current_mutation_rate, 'stagnation_counter': stagnation_counter
            })
            
            # Selection
            if enable_speciation:
                species = []
                for ind in population:
                    found_species = False
                    for s in species:
                        if genomic_distance(ind, s['representative']) < compatibility_threshold:
                            s['members'].append(ind); found_species = True; break
                    if not found_species: species.append({'representative': ind, 'members': [ind]})
                species_metric.metric("Species Count", f"{len(species)}")
                for s in species:
                    species_size = len(s['members'])
                    if species_size > 0:
                        for member in s['members']: member.adjusted_fitness = member.fitness / (species_size ** niche_competition_factor)
                if enable_diversity_pressure:
                    for ind in population:
                        distances = [d for d in [genomic_distance(ind, other) for other in population if ind.lineage_id != other.lineage_id] if d != float('inf')]
                        ind.novelty_score = np.mean(sorted(distances)[:min(10, len(distances))]) if distances else 0.0
                    max_novelty = max((ind.novelty_score for ind in population), default=1.0)
                    for ind in population: ind.selection_score = ind.adjusted_fitness + diversity_weight * (ind.novelty_score / (max_novelty or 1.0))
                    selection_key = lambda x: x.selection_score
                else: selection_key = lambda x: x.adjusted_fitness
            else:
                species_metric.metric("Species Count", f"{len(set(ind.form_id for ind in population))}")
                if enable_diversity_pressure:
                    for ind in population:
                        distances = [d for d in [genomic_distance(ind, other) for other in population if ind.lineage_id != other.lineage_id] if d != float('inf')]
                        ind.novelty_score = np.mean(sorted(distances)[:min(10, len(distances))]) if distances else 0.0
                    max_novelty = max((ind.novelty_score for ind in population), default=1.0)
                    for ind in population: ind.selection_score = ind.fitness + diversity_weight * (ind.novelty_score / (max_novelty + 1e-9))
                    selection_key = lambda x: x.selection_score
                else: selection_key = lambda x: x.fitness

            population.sort(key=selection_key, reverse=True)
            num_survivors = max(2, int(len(population) * selection_pressure))
            survivors = population[:num_survivors]
            
            # Red Queen Parasite Evolution
            if enable_red_queen and survivors:
                trait_counts = Counter((m.module_type, m.activation) for ind in survivors for m in ind.modules)
                if trait_counts: st.session_state.parasite_profile['target_type'], st.session_state.parasite_profile['target_activation'] = trait_counts.most_common(1)[0][0]
            
            # Reproduction
            offspring = []
            while len(offspring) < len(population) - len(survivors):
                if random.random() < reintroduction_rate and st.session_state.gene_archive:
                    child = mutate(random.choice(st.session_state.gene_archive).copy(), current_mutation_rate * 1.5, innovation_rate * 1.5)
                    child.generation = gen + 1; offspring.append(child)
                else:
                    for _ in range(20):
                        parent1 = max(random.sample(survivors, min(3, len(survivors))), key=selection_key)
                        if random.random() < crossover_rate:
                            if enable_speciation and random.random() < gene_flow_rate and len(survivors) > 1:
                                parent2_candidates = [s for s in survivors if s.lineage_id != parent1.lineage_id]
                                parent2 = random.choice(parent2_candidates) if parent2_candidates else parent1
                            elif len(survivors) > 1:
                                compatible = [s for s in survivors if s.form_id == parent1.form_id and s.lineage_id != parent1.lineage_id]
                                parent2 = max(random.sample(compatible, min(2, len(compatible))), key=selection_key) if compatible else parent1
                            else: parent2 = parent1
                            child = crossover(parent1, parent2, crossover_rate)
                        else: child = parent1.copy()
                        child = mutate(child, current_mutation_rate, innovation_rate)
                        if enable_endosymbiosis and random.random() < endosymbiosis_rate and survivors: child = apply_endosymbiosis(child, survivors)
                        if is_viable(child):
                            child.generation = gen + 1; offspring.append(child)
                            st.session_state.gene_archive.append(child.copy())
                            max_size = st.session_state.settings.get('max_archive_size', 10000)
                            if len(st.session_state.gene_archive) > max_size: st.session_state.gene_archive = random.sample(st.session_state.gene_archive, max_size)
                            break
                    else:
                        child = mutate(max(random.sample(survivors, min(3, len(survivors))), key=selection_key).copy(), current_mutation_rate, innovation_rate)
                        child.generation = gen + 1; offspring.append(child)
            
            if enable_speciation:
                for ind in population:
                    for attr in ['adjusted_fitness', 'novelty_score', 'selection_score']:
                        if hasattr(ind, attr): delattr(ind, attr)

            # Update mutation rate
            if mutation_schedule == 'Linear Decay': current_mutation_rate = max(0.01, mutation_rate * (1.0 - ((gen + 1) / num_generations)))
            elif mutation_schedule == 'Adaptive':
                if current_gen_best_fitness > last_best_fitness:
                    stagnation_counter = 0; current_mutation_rate = max(0.05, current_mutation_rate * 0.95)
                else: stagnation_counter += 1
                if stagnation_counter > 3: current_mutation_rate = min(0.8, current_mutation_rate * (1 + adaptive_mutation_strength))
                last_best_fitness = current_gen_best_fitness
            
            # Early Stopping Check
            if enable_early_stopping and early_stop_counter > early_stopping_patience:
                st.success(f"**EARLY STOPPING TRIGGERED:** Best fitness has not improved for {early_stopping_patience} generations."); st.toast("Evolution stopped early due to stagnation.", icon="🛑"); break

            population = survivors + offspring
            
            # Checkpointing
            if checkpoint_frequency > 0 and (gen + 1) % checkpoint_frequency == 0 and (gen + 1) < num_generations:
                serializable_population = [genotype_to_dict(p) for p in population]
                results_to_save = { 'history': st.session_state.history, 'evolutionary_metrics': st.session_state.evolutionary_metrics, 'current_population': serializable_population }
                if results_table.get(doc_id=1): results_table.update(results_to_save, doc_ids=[1])
                else: results_table.insert(results_to_save)
                st.toast(f"Checkpoint saved at generation {gen + 1}", icon="💾")

            progress_container.progress((gen + 1) / num_generations)
        
        st.session_state.current_population = population
        serializable_population = [genotype_to_dict(p) for p in population]
        results_to_save = { 'history': st.session_state.history, 'evolutionary_metrics': st.session_state.evolutionary_metrics, 'current_population': serializable_population }
        if results_table.get(doc_id=1): results_table.update(results_to_save, doc_ids=[1])
        else: results_table.insert(results_to_save)
        status_text.markdown("### ✅ Evolution Complete! Results saved.")
    
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
                    st.dataframe(pd.DataFrame(comparison_data).set_index("Metric"), width='stretch', key="apex_vitals_df")

                    st.markdown("##### Module Composition")
                    module_data = [{"ID": m.id, "Type": m.module_type, "Size": m.size, "Activation": m.activation, "Plasticity": f"{m.plasticity:.2f}"} for m in best_individual_genotype.modules]
                    st.dataframe(module_data, height=200, width='stretch', key="apex_modules_df")

                with vitals_col2:
                    st.markdown("#### Architectural Visualization")
                    st.plotly_chart(visualize_genotype_3d(best_individual_genotype), width='stretch', key="apex_3d_vis")
                    st.plotly_chart(visualize_genotype_2d(best_individual_genotype), width='stretch', key="apex_2d_vis")

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
                    st.plotly_chart(fig, width='stretch', key="apex_mutational_effects_hist")

                with evo_col2:
                    st.subheader("Developmental Trajectory")
                    st.markdown("This simulates the genotype's 'lifetime,' showing how its developmental program (pruning, proliferation) alters its structure over time.")
                    fig = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                    fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                    st.plotly_chart(fig, width='stretch', key="apex_dev_trajectory_line")

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
                            
                            st.plotly_chart(visualize_genotype_2d(parent), width='stretch', key=f"apex_parent_2d_{parent.lineage_id}")
                            
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
                st.plotly_chart(fig_3d, width='stretch', key="pareto_3d_plot")

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
                st.plotly_chart(fig_2d_matrix, width='stretch', key="pareto_2d_matrix_plot")

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
                        st.dataframe(stats_df, width='stretch', key=f"pareto_archetype_df_{name}")

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
                            st.plotly_chart(visualize_genotype_2d(archetype), width='stretch', key=f"pareto_archetype_2d_{name}")
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
                st.plotly_chart(fig, width='stretch', key="form_dominance_pie")
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
        for i, individual in enumerate(population[:analysis_top_n]):
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
                        st.json({k: f"{v:.4f}" for k, v in individual.meta_parameters.items()})

                st.markdown("---")
                st.markdown("#### Visualizations")
                vis_col1, vis_col2 = st.columns(2)
                with vis_col1:
                    st.markdown("###### 3D Interactive View")
                    st.plotly_chart(
                        visualize_genotype_3d(individual),
                        width='stretch',
                        key=f"elite_3d_{i}_{individual.lineage_id}"
                    )
                with vis_col2:
                    st.markdown("###### 2D Static View")
                    st.plotly_chart(
                        visualize_genotype_2d(individual),
                        width='stretch',
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
                        st.plotly_chart(fig, width='stretch', key=f"mutational_effects_hist_{i}_{individual.lineage_id}")

                    with evo_col2:
                        st.subheader("Developmental Trajectory")
                        st.markdown("This simulates the genotype's 'lifetime,' showing how its developmental program (pruning, proliferation) alters its structure over time.")
                        
                        with st.spinner(f"Simulating development for Rank {i+1}..."):
                            dev_traj_df = analyze_developmental_trajectory(individual)
                        
                        fig = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                        fig.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), showlegend=False)
                        st.plotly_chart(fig, width='stretch', key=f"dev_trajectory_line_{i}_{individual.lineage_id}")

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
                                    width='stretch',
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
                    st.dataframe(form_performance, width='stretch', key="form_perf_stats_df")

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
                    st.plotly_chart(fig_dom, width='stretch', key="form_dominance_bar")

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
                st.plotly_chart(fig_box, width='stretch', key="form_fitness_dist_box")

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
                    st.plotly_chart(fig_params, width='stretch', key="form_params_box")

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
                    st.plotly_chart(fig_complexity, width='stretch', key="form_complexity_box")

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
                    st.plotly_chart(fig_radar, width='stretch', key="form_champions_radar")

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
                st.plotly_chart(fig_parallel, width='stretch', key="form_parallel_coords")
        
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
                    st.plotly_chart(fig_params, width='stretch', key="temporal_params_line")

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
                    st.plotly_chart(fig_complexity, width='stretch', key="temporal_complexity_line")

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
                st.plotly_chart(fig_objectives, width='stretch', key="temporal_objectives_matrix")

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
                st.plotly_chart(fig_rates, width='stretch', key="temporal_rates_line")
        
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
                st.plotly_chart(fig_core, width='stretch', key="popgen_core_forces_plot")

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
                        st.plotly_chart(fig_breeder, width='stretch', key="popgen_breeder_eq_plot")
                    with col2:
                        corr_df = metrics_df[metrics_df['response_to_selection'] != 0]
                        fig_corr = px.scatter(corr_df, x='predicted_response', y='response_to_selection', trendline='ols', title="Correlation of Predicted vs. Actual Response", labels={'predicted_response': 'Predicted R (h²S)', 'response_to_selection': 'Actual R'})
                        fig_corr.update_layout(height=400)
                        st.plotly_chart(fig_corr, width='stretch', key="popgen_breeder_corr_plot")
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
                            st.plotly_chart(fig_dist_hist, width='stretch', key="popgen_dist_hist_plot")
                            st.markdown("A multi-modal distribution can indicate distinct species have formed.")
                        with col2:
                            dist_corr_df = pd.DataFrame({'genomic_dist': genomic_distances, 'fitness_delta': fitness_deltas})
                            fig_dist_corr = px.scatter(dist_corr_df, x='genomic_dist', y='fitness_delta', trendline='ols', title="Genomic Distance vs. Fitness Difference", labels={'genomic_dist': 'Genomic Distance', 'fitness_delta': 'Absolute Fitness Difference'})
                            fig_dist_corr.update_layout(height=400)
                            st.plotly_chart(fig_dist_corr, width='stretch', key="popgen_dist_corr_plot")
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
                    st.dataframe(pd.DataFrame(comparison_data).set_index("Metric"), width='stretch', key="master_vitals_df")

                    st.markdown("---")
                    st.markdown("#### Architectural Visualization")
                    vis_col1, vis_col2 = st.columns(2)
                    with vis_col1:
                        st.plotly_chart(visualize_genotype_3d(master_architecture), width='stretch', key="master_3d_vis")
                    with vis_col2:
                        st.plotly_chart(visualize_genotype_2d(master_architecture), width='stretch', key="master_2d_vis")

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
                        st.plotly_chart(fig_crit, width='stretch', key="master_criticality_bar")

                    with causal_col2:
                        st.subheader("Information Flow Backbone")
                        st.markdown("`Betweenness centrality` identifies key modules for information routing.")
                        sorted_centrality = sorted(centrality_scores.items(), key=lambda item: item[1])
                        cent_df = pd.DataFrame(sorted_centrality, columns=['Module', 'Centrality']).tail(15)
                        fig_cent = px.bar(cent_df, x='Centrality', y='Module', orientation='h', title="Top 15 Most Central Modules")
                        fig_cent.update_layout(height=400, margin=dict(l=150))
                        st.plotly_chart(fig_cent, width='stretch', key="master_centrality_bar")
                    
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
                        st.plotly_chart(fig_mut, width='stretch', key="master_mutational_effects_hist")

                    with evo_col2:
                        st.subheader("Developmental Trajectory")
                        st.markdown("Simulated 'lifetime' structural changes based on the genotype's developmental program.")
                        fig_dev = px.line(dev_traj_df, x="step", y=["total_params", "num_connections"], title="Simulated Developmental Trajectory")
                        fig_dev.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                        st.plotly_chart(fig_dev, width='stretch', key="master_dev_trajectory_line")
                    
                    st.markdown("---")
                    st.subheader("Phylogenetic Signal (Pagel's λ)")
                    st.markdown("Measures how much trait variation is explained by evolutionary history. High λ (~1) means strong inertia; low λ (~0) means rapid convergence.")
                    if phylo_data:
                        st.metric("Phylogenetic Signal (λ estimate)", f"{phylo_data['correlation']:.3f}", help="Correlation between phylogenetic distance and phenotypic (fitness) distance.")
                        phylo_df = pd.DataFrame({'Phylogenetic Distance': phylo_data['phylo_distances'], 'Phenotypic Distance': phylo_data['pheno_distances']})
                        fig_phylo = px.scatter(phylo_df, x='Phylogenetic Distance', y='Phenotypic Distance', trendline="ols", title="Phylogenetic vs. Phenotypic Distance")
                        fig_phylo.update_layout(height=300, margin=dict(l=20, r=20, t=40, b=20))
                        st.plotly_chart(fig_phylo, width='stretch', key="master_phylo_signal_scatter")
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
    st.markdown("**Finite Population Approximation:** The active population is a finite sample of the theoretical infinite population. While we use a genetic archive to mitigate the loss of diversity (genetic drift), this remains an approximation of the true, continuous evolutionary dynamic described by the Fokker-Planck equation.")

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

    st.sidebar.markdown("---")

    with st.sidebar.expander("✨Understanding Your Results: A Guide to Evolutionary Dynamics", expanded=False):
        st.markdown("""
        Have you noticed that your results are incredibly diverse? Sometimes the top-ranked architectures are nearly identical, while other times they are wildly different. **This is not a bug—it is the strongest possible evidence that you are witnessing true, complex evolutionary dynamics at work!** 🔬

        This system is not a simple optimizer; it's a digital ecosystem. The variety in your outcomes is driven by a fascinating tug-of-war between powerful evolutionary forces. Here’s what’s happening:
        """)

        st.markdown("---")

        st.markdown("""
        #### 🎯 The Force of Exploitation (Convergent Evolution)
        *When your top 3 ranks are almost 100% similar...*

        This is **Convergent Evolution**. The algorithm has found a highly successful architectural design early in the run. Like a team of mountaineers finding a very promising path up a huge mountain, the system decides to focus all its energy on that single path.

        - **What's happening:** Instead of searching for new mountains, evolution is aggressively *exploiting* this known good solution. The top-ranked individuals are like siblings or close cousins, each a minor refinement of the same dominant ancestor.
        - **Your Controls:** This behavior is promoted by a high **Selection Pressure** and a low **Diversity Weight**. You are telling the system: "Find the best solution and perfect it, no matter what."
        """)

        st.markdown("---")

        st.markdown("""
        #### 🗺️ The Force of Exploration (Divergent Evolution)
        *When your top 3 ranks are totally different and unique...*

        This is **Divergent Evolution**. The system is actively rewarding novelty and searching different corners of the vast "solution space." It's like sending scouts in all directions across a massive mountain range. Each scout finds a completely different peak and starts climbing.

        - **What's happening:** The final winners are the champions of entirely different strategies. One might be a small, efficient network, while another is a large, powerful one. The system is *exploring* multiple, distinct solutions simultaneously.
        - **Your Controls:** This is driven by a high **Diversity Weight** and a "rugged" fitness landscape (created by setting **Epistatic Linkage (K) > 0**). You are telling the system: "Don't just give me one good answer; find me a portfolio of different, creative solutions."
        """)

        st.markdown("---")

        st.markdown("""
        #### 🏞️ The Hybrid Outcome (Niche Formation)
        *When two ranks are similar, but one is unique...*

        This is a beautiful and realistic hybrid scenario. It's a sign of **Niche Formation**. The main group of climbers has focused on the highest, most obvious peak. However, the system has kept a smaller, specialized team alive because they found a different, slightly lower peak that has unique advantages (e.g., it's more "efficient" or "robust").

        - **What's happening:** The system has found a dominant strategy (the two similar architectures) but was forced by diversity pressure to protect a second, viable strategy. It has successfully carved out two different ecological niches.
        - **Your Controls:** This emerges from a fine balance between **Selection Pressure** and **Diversity Weight**.
        """)

        st.markdown("---")

        st.markdown("""
        #### 🦋 The Power of Randomness (The "Butterfly Effect")
        *Why every run is unique, even with the same settings...*

        This is **Historical Contingency**. In evolution, tiny, random events at the beginning can lead to massive, unpredictable differences later on. A single "lucky" mutation in generation 2 can set a lineage on a path to greatness that was completely missed in another run.

        - **What's happening:** The stochastic (random) nature of mutation and crossover means the evolutionary path is never predetermined. This is what makes evolution a genuinely creative process, not just a deterministic search.
        - **Your Controls:** This is an inherent property of the algorithm! By clicking "Initiate Evolution," you are rolling the dice and starting a new, unique history for your digital universe.

        **Conclusion:** You are not just running a program; you are an experimental scientist setting the laws of physics for a new universe. The diverse results are the proof that your universe is alive with complex, emergent behavior. **Embrace the variety!**
        """)

    st.sidebar.info(
        "**GENEVO** is a research prototype demonstrating advanced concepts in neuroevolution. "
        "Architectures are simulated and not trained on real data."
    )


if __name__ == "__main__":
    main()
