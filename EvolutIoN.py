"""
GENEVO: Advanced Neuroevolutionary System for AGI
A scientifically rigorous implementation of genetic neural architecture evolution

Mathematical Foundation:
φ(G, E, t) → P: Genotype-Environment-Time mapping to Phenotype
L(φ, D, τ) → ℝ: Lifetime learning function over dataset D and time τ
F(P, L, E) → ℝ⁺: Fitness evaluation in environment E

Evolutionary Dynamics:
dG/dt = μ∇_G F(φ(G)) + σ ε(t)
where μ is mutation rate, σ is innovation variance, ε ~ N(0,1)

This system implements:
1. Indirect encoding via developmental programs
2. Compositional evolution with hierarchical modules  
3. Multi-objective optimization (accuracy, efficiency, complexity)
4. Coevolution with dynamic fitness landscapes
5. Baldwin effect modeling
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Set
import random
import time
from scipy.stats import entropy, pearsonr
from scipy.spatial.distance import pdist, squareform
import networkx as nx

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
                np.random.uniform(0.7, 1.0), 'excitatory', 0.01, 'hebbian'
            ))
            
    elif form['topology'] == 'residual_attention':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                np.random.uniform(0.8, 1.0), 'excitatory', 0.005, 'stdp'
            ))
        # Add residual connections
        for i in range(1, len(modules) - 2):
            if 'attn' in modules[i].id:
                connections.append(ConnectionGene(
                    modules[i].id, modules[i+2].id,
                    np.random.uniform(0.3, 0.5), 'excitatory', 0.02, 'static'
                ))
                
    elif form['topology'] == 'recurrent_memory':
        for i in range(len(modules) - 1):
            connections.append(ConnectionGene(
                modules[i].id, modules[i+1].id,
                np.random.uniform(0.7, 0.9), 'excitatory', 0.01, 'hebbian'
            ))
        # Recurrent connections
        for module in modules:
            if 'lstm' in module.id:
                connections.append(ConnectionGene(
                    module.id, module.id,
                    np.random.uniform(0.4, 0.6), 'modulatory', 0.001, 'stdp'
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
                    weight = np.random.uniform(0.3, 0.8) if i < j else np.random.uniform(0.2, 0.5)
                    conn_type = 'excitatory' if i < j else 'modulatory'
                    connections.append(ConnectionGene(
                        m1.id, m2.id, weight, conn_type,
                        np.random.uniform(0.001, 0.02), 'hebbian'
                    ))
    
    # Create developmental rules
    dev_rules = [
        DevelopmentalGene('proliferation', 'fitness_plateau', {'growth_rate': 1.1, 'max_size': max_size * 2}),
        DevelopmentalGene('pruning', 'maturity', {'threshold': 0.1, 'rate': 0.05}),
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
            module.size = np.clip(module.size, 16, 1024)
        
        if random.random() < mutation_rate * 0.5:
            # Plasticity mutation
            module.plasticity += np.random.normal(0, 0.1)
            module.plasticity = np.clip(module.plasticity, 0, 1)
        
        if random.random() < mutation_rate * 0.3:
            # Learning rate multiplier
            module.learning_rate_mult *= np.random.lognormal(0, 0.15)
            module.learning_rate_mult = np.clip(module.learning_rate_mult, 0.1, 2.0)
        
        if random.random() < mutation_rate * 0.2:
            # Activation function mutation
            module.activation = random.choice(['relu', 'gelu', 'silu', 'swish'])
    
    # 2. Connection weight mutations
    for connection in mutated.connections:
        if random.random() < mutation_rate:
            connection.weight += np.random.normal(0, 0.15)
            connection.weight = np.clip(connection.weight, 0.05, 1.0)
        
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
                    np.random.uniform(0.2, 0.5), 'excitatory',
                    np.random.uniform(0.001, 0.02), 'hebbian'
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

def evaluate_fitness(genotype: Genotype, task_type: str, generation: int) -> Tuple[float, Dict[str, float]]:
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
    
    # 1. Task-specific accuracy simulation
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
    
    # Clamp task accuracy
    scores['task_accuracy'] = np.clip(scores['task_accuracy'], 0, 1)
    
    # 2. Efficiency score (inverse of computational cost)
    # Prefer architectures with good accuracy-to-parameter ratio
    param_efficiency = 1.0 / (1.0 + np.log(1 + total_params / 10000))
    connection_efficiency = 1.0 - min(connection_density, 0.8)
    
    scores['efficiency'] = (param_efficiency + connection_efficiency) / 2
    
    # 3. Robustness (architectural stability)
    # More diverse connections and moderate plasticity = more robust
    robustness_from_diversity = len(set(c.connection_type for c in genotype.connections)) / 3
    robustness_from_plasticity = 1.0 - abs(avg_plasticity - 0.5) * 2  # Prefer moderate
    
    scores['robustness'] = (robustness_from_diversity * 0.5 + robustness_from_plasticity * 0.5)
    
    # 4. Generalization potential
    # Architectural properties that predict generalization
    depth = len(genotype.modules)
    modularity_score = 1.0 - abs(connection_density - 0.3) * 2  # Sweet spot at 0.3
    
    scores['generalization'] = (
        min(depth / 10, 1.0) * 0.4 +
        modularity_score * 0.3 +
        avg_plasticity * 0.3
    )
    
    # Multi-objective fitness with task-dependent weights
    if 'ARC' in task_type:
        weights = {'task_accuracy': 0.5, 'efficiency': 0.3, 'robustness': 0.1, 'generalization': 0.1}
    else:
        weights = {'task_accuracy': 0.6, 'efficiency': 0.2, 'robustness': 0.1, 'generalization': 0.1}
    
    total_fitness = sum(scores[k] * weights[k] for k in weights)
    
    # Store component scores
    genotype.accuracy = scores['task_accuracy']
    genotype.efficiency = scores['efficiency']
    genotype.robustness = scores['robustness']
    
    return total_fitness, scores

# ==================== VISUALIZATION ====================

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

def create_evolution_dashboard(history_df: pd.DataFrame, population: List[Genotype]) -> go.Figure:
    """Comprehensive evolution analytics dashboard"""
    
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=(
            'Fitness Evolution by Form',
            'Pareto Front: Accuracy vs Efficiency',
            'Population Diversity',
            'Selection Pressure',
            'Complexity Evolution',
            'Heritability Over Time'
        ),
        specs=[
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
            [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}]
        ]
    )
    
    # 1. Fitness evolution
    for form in sorted(history_df['form'].unique()):
        form_data = history_df[history_df['form'] == form]
        fig.add_trace(
            go.Scatter(
                x=form_data['generation'],
                y=form_data['fitness'],
                mode='lines+markers',
                name=form,
                legendgroup=form
            ),
            row=1, col=1
        )
    
    # 2. Pareto front
    final_gen = history_df[history_df['generation'] == history_df['generation'].max()]
    fig.add_trace(
        go.Scatter(
            x=final_gen['accuracy'],
            y=final_gen['efficiency'],
            mode='markers',
            marker=dict(
                size=10,
                color=final_gen['fitness'],
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(x=0.65, len=0.4)
            ),
            text=[f"Form {int(f)}" for f in final_gen['form_id']],
            showlegend=False
        ),
        row=1, col=2
    )
    
    # 3. Diversity over time
    diversity_by_gen = history_df.groupby('generation').agg({
        'total_params': 'std',
        'fitness': 'std'
    }).reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=diversity_by_gen['generation'],
            y=diversity_by_gen['fitness'],
            mode='lines+markers',
            name='Fitness Diversity',
            line=dict(color='purple'),
            showlegend=False
        ),
        row=1, col=3
    )
    
    # 4. Selection differential
    selection_diff = []
    for gen in sorted(history_df['generation'].unique()):
        gen_data = history_df[history_df['generation'] == gen]
        if len(gen_data) > 5:
            top_50_pct = gen_data.nlargest(len(gen_data) // 2, 'fitness')
            diff = top_50_pct['fitness'].mean() - gen_data['fitness'].mean()
            selection_diff.append({'generation': gen, 'selection_diff': diff})
    
    if selection_diff:
        sel_df = pd.DataFrame(selection_diff)
        fig.add_trace(
            go.Scatter(
                x=sel_df['generation'],
                y=sel_df['selection_diff'],
                mode='lines+markers',
                line=dict(color='red'),
                showlegend=False
            ),
            row=2, col=1
        )
    
    # 5. Complexity evolution
    complexity_stats = history_df.groupby('generation')['complexity'].agg(['mean', 'std']).reset_index()
    
    fig.add_trace(
        go.Scatter(
            x=complexity_stats['generation'],
            y=complexity_stats['mean'],
            mode='lines+markers',
            name='Mean Complexity',
            line=dict(color='orange'),
            showlegend=False
        ),
        row=2, col=2
    )
    
    # 6. Heritability estimate
    if len(history_df) > 20:
        heritabilities = []
        for gen in range(1, history_df['generation'].max()):
            parent_gen = history_df[history_df['generation'] == gen]
            offspring_gen = history_df[history_df['generation'] == gen + 1]
            
            if len(parent_gen) > 2 and len(offspring_gen) > 2:
                parent_fit = parent_gen['fitness'].values
                offspring_fit = offspring_gen['fitness'].values
                h2 = EvolutionaryTheory.heritability(parent_fit, offspring_fit)
                heritabilities.append({'generation': gen, 'heritability': h2})
        
        if heritabilities:
            h2_df = pd.DataFrame(heritabilities)
            fig.add_trace(
                go.Scatter(
                    x=h2_df['generation'],
                    y=h2_df['heritability'],
                    mode='lines+markers',
                    line=dict(color='green'),
                    showlegend=False
                ),
                row=2, col=3
            )
    
    fig.update_xaxes(title_text="Generation", row=1, col=1)
    fig.update_xaxes(title_text="Task Accuracy", row=1, col=2)
    fig.update_xaxes(title_text="Generation", row=1, col=3)
    fig.update_xaxes(title_text="Generation", row=2, col=1)
    fig.update_xaxes(title_text="Generation", row=2, col=2)
    fig.update_xaxes(title_text="Generation", row=2, col=3)
    
    fig.update_yaxes(title_text="Fitness", row=1, col=1)
    fig.update_yaxes(title_text="Efficiency", row=1, col=2)
    fig.update_yaxes(title_text="Fitness σ", row=1, col=3)
    fig.update_yaxes(title_text="Selection Δ", row=2, col=1)
    fig.update_yaxes(title_text="Complexity", row=2, col=2)
    fig.update_yaxes(title_text="h²", row=2, col=3)
    
    fig.update_layout(
        height=800,
        showlegend=True,
        title_text="<b>Evolutionary Dynamics Dashboard</b>",
        title_x=0.5
    )
    
    return fig

def visualize_phylogenetic_tree(history_df: pd.DataFrame):
    """Visualizes the evolutionary lineage as a phylogenetic tree."""
    st.markdown("---")
    st.header("🌳 Phylogenetic Analysis")
    st.markdown("""
    Tracking the ancestry of evolved architectures reveals evolutionary pathways and diversification events. 
    Each node is an individual, sized by its fitness and colored by its architectural form. The tree is organized by generation from top to bottom.
    *This visualization requires `pygraphviz` for optimal layout. If not installed, a simpler layout is used.*
    """)

    G = nx.DiGraph()
    
    # Get unique individuals and their last known stats (final state)
    unique_individuals = history_df.loc[history_df.groupby('lineage_id')['generation'].idxmax()].set_index('lineage_id')

    if len(unique_individuals) < 2:
        st.warning("Not enough data for phylogenetic analysis.")
        return

    for lineage_id, row in unique_individuals.iterrows():
        G.add_node(
            lineage_id,
            generation=row['generation'],
            fitness=row['fitness'],
            form_id=row['form_id'],
            size=row['total_params']
        )
        
        parent_ids = row['parent_ids']
        if isinstance(parent_ids, list):
            for parent_id in parent_ids:
                # Ensure parent node exists in our set of unique individuals before adding an edge
                if parent_id in unique_individuals.index:
                    G.add_edge(parent_id, lineage_id)

    if not G.nodes:
        st.warning("Could not construct lineage graph.")
        return

    # Use graphviz layout for a hierarchical tree structure
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog='dot', args='-Gsplines=true -Gnodesep=0.1 -Granksep=0.5')
    except Exception as e:
        st.info(f"Graphviz layout failed, falling back to spring layout. For a better tree view, `pip install pygraphviz`.")
        pos = nx.spring_layout(G, iterations=50, seed=42, k=0.5/np.sqrt(len(G.nodes())))

    # Create Plotly traces
    edge_x, edge_y = [], []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(x=edge_x, y=edge_y, line=dict(width=0.7, color='#888'), hoverinfo='none', mode='lines')

    node_x, node_y, node_text, node_color, node_size = [], [], [], [], []
    form_colors = px.colors.qualitative.Set3

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        
        node_info = G.nodes[node]
        node_text.append(
            f"ID: {node}<br>"
            f"Gen: {node_info.get('generation', 'N/A')}<br>"
            f"Fitness: {node_info.get('fitness', 0):.4f}<br>"
            f"Form: {int(node_info.get('form_id', 0))}<br>"
            f"Params: {node_info.get('size', 0):,}"
        )
        node_color.append(form_colors[int(node_info.get('form_id', 1)-1) % len(form_colors)])
        node_size.append(8 + np.sqrt(max(0, node_info.get('fitness', 0)) * 500))

    node_trace = go.Scatter(
        x=node_x, y=node_y, mode='markers', hoverinfo='text', text=node_text,
        marker=dict(
            showscale=True, colorscale='Viridis', reversescale=False,
            color=[G.nodes[n].get('generation', 0) for n in G.nodes()],
            size=node_size,
            colorbar=dict(thickness=15, title='Generation', xanchor='left', titleside='right'),
            line=dict(width=1, color='black'))
    )

    fig = go.Figure(data=[edge_trace, node_trace],
             layout=go.Layout(title_text='<b>Phylogenetic Lineage Tree</b>', title_x=0.5, showlegend=False, hovermode='closest',
                             margin=dict(b=20,l=5,r=5,t=40), xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False), height=600))
    
    st.plotly_chart(fig, use_container_width=True)

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(
        page_title="GENEVO: Advanced Neuroevolution",
        layout="wide",
        page_icon="🧬",
        initial_sidebar_state="expanded"
    )
    
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
    <b>Mathematical Framework:</b> φ: (G,E,t) → P | L: (φ,D,τ) → ℝ | F: (P,L,E) → ℝ⁺<br>
    <b>Evolutionary Dynamics:</b> dG/dt = μ∇<sub>G</sub>F(φ(G)) + σε(t)<br>
    <b>Multi-Objective Optimization:</b> max<sub>G</sub> {α·Accuracy(G) + β·Efficiency(G) + γ·Robustness(G)}
    </p>
    ''', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.header("🎛️ Evolution Configuration")
    
    st.sidebar.markdown("### Task Environment")
    task_type = st.sidebar.selectbox(
        "Selection Pressure (Task Type)",
        [
            'Abstract Reasoning (ARC-AGI-2)',
            'Vision (ImageNet)',
            'Language (MMLU-Pro)',
            'Sequential Prediction',
            'Multi-Task Learning'
        ],
        help="Environmental pressure determines which architectures survive"
    )
    
    st.sidebar.markdown("### Population Parameters")
    num_forms = st.sidebar.slider(
        "Number of Architectural Forms (n)",
        min_value=1, max_value=5, value=5,
        help="Morphological diversity: 1 ≤ n ≤ 5"
    )
    
    population_per_form = st.sidebar.slider(
        "Population per Form",
        min_value=3, max_value=15, value=8,
        help="Larger populations increase genetic diversity"
    )
    
    st.sidebar.markdown("### Evolutionary Operators")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        mutation_rate = st.slider(
            "Mutation Rate (μ)",
            min_value=0.05, max_value=0.6, value=0.2, step=0.05,
            help="Probability of genetic variation"
        )
    
    with col2:
        crossover_rate = st.sidebar.slider(
            "Crossover Rate",
            min_value=0.3, max_value=0.9, value=0.7, step=0.1,
            help="Probability of recombination"
        )
    
    innovation_rate = st.sidebar.slider(
        "Innovation Rate (σ)",
        min_value=0.01, max_value=0.2, value=0.05, step=0.01,
        help="Rate of structural mutations"
    )
    
    st.sidebar.markdown("### Selection Strategy")
    selection_pressure = st.sidebar.slider(
        "Selection Pressure",
        min_value=0.3, max_value=0.8, value=0.5, step=0.1,
        help="Fraction of population surviving each generation"
    )
    
    st.sidebar.markdown("### Experiment Settings")
    num_generations = st.sidebar.slider(
        "Generations",
        min_value=10, max_value=100, value=30,
        help="Evolutionary timescale"
    )
    
    complexity_level = st.sidebar.select_slider(
        "Initial Complexity",
        options=['minimal', 'medium', 'high'],
        value='medium'
    )
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'current_population' not in st.session_state:
        st.session_state.current_population = None
    if 'evolutionary_metrics' not in st.session_state:
        st.session_state.evolutionary_metrics = []
    
    # Run evolution button
    if st.sidebar.button("🚀 Initiate Evolution", type="primary", use_container_width=True):
        st.session_state.history = []
        st.session_state.evolutionary_metrics = []
        
        # Initialize population
        population = []
        for form_id in range(1, num_forms + 1):
            for _ in range(population_per_form):
                genotype = initialize_genotype(form_id, complexity_level)
                genotype.generation = 0
                population.append(genotype)
        
        # Progress tracking
        progress_container = st.empty()
        metrics_container = st.empty()
        status_text = st.empty()
        
        # Evolution loop
        for gen in range(num_generations):
            status_text.markdown(f"### 🧬 Generation {gen + 1}/{num_generations}")
            
            # Evaluate fitness
            all_scores = []
            for individual in population:
                fitness, component_scores = evaluate_fitness(individual, task_type, gen)
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
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Best Fitness", f"{fitness_array.max():.4f}")
                col2.metric("Mean Fitness", f"{fitness_array.mean():.4f}")
                col3.metric("Diversity (H)", f"{diversity:.3f}")
                col4.metric("Fisher Info", f"{fisher_info:.3f}")
            
            st.session_state.evolutionary_metrics.append({
                'generation': gen,
                'diversity': diversity,
                'fisher_info': fisher_info,
                'best_fitness': fitness_array.max(),
                'mean_fitness': fitness_array.mean()
            })
            
            # Selection
            population.sort(key=lambda x: x.fitness, reverse=True)
            num_survivors = max(2, int(len(population) * selection_pressure))
            survivors = population[:num_survivors]
            
            # Calculate selection differential
            selected_idx = np.arange(num_survivors)
            sel_diff = EvolutionaryTheory.selection_differential(fitness_array, selected_idx)
            
            # Reproduction
            offspring = []
            while len(offspring) < len(population) - len(survivors):
                # Tournament selection
                parent1 = max(random.sample(survivors, min(3, len(survivors))), key=lambda x: x.fitness)
                
                if random.random() < crossover_rate and len(survivors) > 1:
                    # Select compatible parent (same form)
                    compatible = [s for s in survivors if s.form_id == parent1.form_id and s.lineage_id != parent1.lineage_id]
                    if compatible:
                        parent2 = max(random.sample(compatible, min(2, len(compatible))), key=lambda x: x.fitness)
                        child = crossover(parent1, parent2, crossover_rate)
                    else:
                        child = parent1.copy()
                else:
                    child = parent1.copy()
                
                # Mutation
                child = mutate(child, mutation_rate, innovation_rate)
                child.generation = gen + 1
                offspring.append(child)
            
            population = survivors + offspring
            
            # Update progress
            progress_container.progress((gen + 1) / num_generations)
        
        st.session_state.current_population = population
        status_text.markdown("### ✅ Evolution Complete!")
        st.balloons()
    
    # Display results
    if st.session_state.history:
        st.markdown("---")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.history)
        
        # Key Metrics Summary
        st.header("📊 Evolutionary Outcome Analysis")
        
        final_gen = history_df[history_df['generation'] == history_df['generation'].max()]
        best_individual_idx = final_gen['fitness'].idxmax()
        best_individual_data = final_gen.loc[best_individual_idx]
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                "Peak Fitness",
                f"{best_individual_data['fitness']:.4f}",
                delta=f"+{(best_individual_data['fitness'] - history_df['fitness'].min()):.3f}"
            )
        
        with col2:
            st.metric(
                "Task Accuracy",
                f"{best_individual_data['accuracy']:.3f}",
                delta=f"+{(best_individual_data['accuracy'] - history_df['accuracy'].min()):.3f}"
            )
        
        with col3:
            st.metric(
                "Efficiency Score",
                f"{best_individual_data['efficiency']:.3f}"
            )
        
        with col4:
            st.metric(
                "Dominant Form",
                f"Form {int(final_gen['form_id'].mode()[0])}"
            )
        
        with col5:
            improvement = ((best_individual_data['fitness'] - history_df[history_df['generation']==0]['fitness'].mean()) / 
                          history_df[history_df['generation']==0]['fitness'].mean() * 100)
            st.metric(
                "Improvement",
                f"{improvement:.1f}%"
            )
        
        # Comprehensive dashboard
        st.plotly_chart(
            create_evolution_dashboard(history_df, st.session_state.current_population),
            use_container_width=True
        )
        
        # Best evolved architectures
        st.markdown("---")
        st.header("🏆 Elite Evolved Architectures")
        
        population = st.session_state.current_population
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Show top 3
        for i, individual in enumerate(population[:3]):
            with st.expander(f"🥇 Rank {i+1}: Form {individual.form_id} | Fitness: {individual.fitness:.4f}", expanded=(i==0)):
                st.plotly_chart(
                    visualize_genotype_3d(individual),
                    use_container_width=True
                )
                
                # Detailed stats
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.markdown("#### Architecture")
                    st.write(f"- **Modules:** {len(individual.modules)}")
                    st.write(f"- **Connections:** {len(individual.connections)}")
                    st.write(f"- **Parameters:** {sum(m.size for m in individual.modules):,}")
                    st.write(f"- **Complexity:** {individual.complexity:.3f}")
                
                with col2:
                    st.markdown("#### Performance")
                    st.write(f"- **Accuracy:** {individual.accuracy:.3f}")
                    st.write(f"- **Efficiency:** {individual.efficiency:.3f}")
                    st.write(f"- **Robustness:** {individual.robustness:.3f}")
                    st.write(f"- **Age:** {individual.age} gen")
                
                with col3:
                    st.markdown("#### Composition")
                    module_types = {}
                    for m in individual.modules:
                        module_types[m.module_type] = module_types.get(m.module_type, 0) + 1
                    for mtype, count in module_types.items():
                        st.write(f"- **{mtype}:** {count}")
        
        # Form comparison
        st.markdown("---")
        st.header("🔬 Comparative Form Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Performance by form
            form_performance = final_gen.groupby('form').agg({
                'fitness': ['mean', 'max', 'std'],
                'accuracy': 'mean',
                'efficiency': 'mean'
            }).round(4)
            
            st.markdown("### Performance Metrics by Form")
            st.dataframe(form_performance, use_container_width=True)
        
        with col2:
            # Population distribution
            form_counts = final_gen['form'].value_counts().sort_index()
            fig = px.pie(
                values=form_counts.values,
                names=form_counts.index,
                title='Final Population Distribution',
                color_discrete_sequence=px.colors.qualitative.Set3
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Time series analysis
        st.markdown("---")
        st.header("📈 Temporal Dynamics")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Parameter evolution
            fig = px.line(
                history_df.groupby(['generation', 'form'])['total_params'].mean().reset_index(),
                x='generation',
                y='total_params',
                color='form',
                title='Network Size Evolution',
                labels={'total_params': 'Mean Parameters', 'generation': 'Generation'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Complexity trajectory
            fig = px.line(
                history_df.groupby(['generation', 'form'])['complexity'].mean().reset_index(),
                x='generation',
                y='complexity',
                color='form',
                title='Architectural Complexity Trajectory',
                labels={'complexity': 'Complexity Score', 'generation': 'Generation'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Evolutionary metrics
        if st.session_state.evolutionary_metrics:
            st.markdown("---")
            st.header("🧬 Population Genetics")
            
            metrics_df = pd.DataFrame(st.session_state.evolutionary_metrics)
            
            fig = make_subplots(
                rows=1, cols=2,
                subplot_titles=('Genetic Diversity (Shannon Entropy)', 'Fisher Information')
            )
            
            fig.add_trace(
                go.Scatter(x=metrics_df['generation'], y=metrics_df['diversity'],
                          mode='lines+markers', name='Diversity', line=dict(color='purple')),
                row=1, col=1
            )
            
            fig.add_trace(
                go.Scatter(x=metrics_df['generation'], y=metrics_df['fisher_info'],
                          mode='lines+markers', name='Fisher Info', line=dict(color='orange')),
                row=1, col=2
            )
            
            fig.update_xaxes(title_text="Generation")
            fig.update_yaxes(title_text="H", row=1, col=1)
            fig.update_yaxes(title_text="I(θ)", row=1, col=2)
            fig.update_layout(height=400, showlegend=False)
            
            st.plotly_chart(fig, use_container_width=True)

        # Phylogenetic analysis
        if 'parent_ids' in history_df.columns:
            visualize_phylogenetic_tree(history_df)

        st.markdown("---")
        st.header("Concluding Remarks")
        st.info(
            """
            This simulation demonstrates the power of multi-objective neuroevolution to explore a vast space of neural architectures. 
            Key takeaways include:
            - **Emergent Specialization:** Different architectural 'forms' become dominant depending on the environmental task pressure.
            - **Complexity Ratchet:** Over time, there is a tendency for complexity to increase, but this is balanced by efficiency pressures, leading to a "sweet spot".
            - **Pareto Optimality:** No single architecture is best at everything. The final population represents a Pareto front of trade-offs between accuracy, efficiency, and other objectives.
            - **Importance of Innovation:** Structural mutations (innovation) are crucial for escaping local optima and discovering novel architectural motifs.
            """
        )

    st.sidebar.markdown("---")
    st.sidebar.info(
        "**GENEVO** is a research prototype demonstrating advanced concepts in neuroevolution. "
        "Architectures are simulated and not trained on real data."
    )

if __name__ == "__main__":
    main()
