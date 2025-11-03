"""
GENEVO Evolution Demonstration
A TRUE SELF-EVOLVING ENTITY for AI Architecture

Formula: X = f(n), where n is number of forms it can adapt into (1 <= n <= 5)

This demo shows how a genetic architecture evolves in real-time, 
adapting its form based on environmental pressure.
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dataclasses import dataclass
from typing import List, Dict, Tuple
import random
import time

# ==================== GENOTYPE DEFINITION ====================

@dataclass
class ModuleGene:
    """A gene encoding a neural module"""
    id: str
    module_type: str  # 'conv', 'attention', 'mlp', 'recurrent', 'graph'
    size: int  # Hidden dimension
    color: str  # For visualization
    
@dataclass
class ConnectionGene:
    """A gene encoding a connection between modules"""
    source: str
    target: str
    weight: float
    
@dataclass
class Genotype:
    """Complete genetic encoding of architecture"""
    modules: List[ModuleGene]
    connections: List[ConnectionGene]
    fitness: float = 0.0
    generation: int = 0
    form_id: int = 1  # Which form (1-5) this genotype represents
    
    def copy(self):
        """Deep copy of genotype"""
        return Genotype(
            modules=[ModuleGene(m.id, m.module_type, m.size, m.color) for m in self.modules],
            connections=[ConnectionGene(c.source, c.target, c.weight) for c in self.connections],
            fitness=self.fitness,
            generation=self.generation,
            form_id=self.form_id
        )

# ==================== EVOLUTION FUNCTIONS ====================

def initialize_genotype(form_id: int) -> Genotype:
    """Initialize a random genotype for a specific form"""
    
    # Define 5 different architectural forms
    forms = {
        1: {  # Simple Sequential (Traditional CNN-like)
            'name': 'Sequential Form',
            'modules': [
                ModuleGene('input', 'conv', 64, '#FF6B6B'),
                ModuleGene('hidden1', 'conv', 128, '#4ECDC4'),
                ModuleGene('hidden2', 'mlp', 256, '#45B7D1'),
                ModuleGene('output', 'mlp', 10, '#96CEB4'),
            ],
            'topology': 'sequential'
        },
        2: {  # Attention-Based (Transformer-like)
            'name': 'Attention Form',
            'modules': [
                ModuleGene('input', 'mlp', 128, '#FF6B6B'),
                ModuleGene('attn1', 'attention', 256, '#FECA57'),
                ModuleGene('attn2', 'attention', 256, '#48DBFB'),
                ModuleGene('output', 'mlp', 10, '#96CEB4'),
            ],
            'topology': 'sequential'
        },
        3: {  # Recurrent (RNN-like)
            'name': 'Recurrent Form',
            'modules': [
                ModuleGene('input', 'mlp', 64, '#FF6B6B'),
                ModuleGene('recurrent1', 'recurrent', 128, '#A29BFE'),
                ModuleGene('recurrent2', 'recurrent', 128, '#6C5CE7'),
                ModuleGene('output', 'mlp', 10, '#96CEB4'),
            ],
            'topology': 'recurrent'
        },
        4: {  # Hybrid Parallel (ResNet-like)
            'name': 'Hybrid Parallel Form',
            'modules': [
                ModuleGene('input', 'conv', 64, '#FF6B6B'),
                ModuleGene('branch1', 'conv', 128, '#FD79A8'),
                ModuleGene('branch2', 'attention', 128, '#FDCB6E'),
                ModuleGene('merge', 'mlp', 256, '#00B894'),
                ModuleGene('output', 'mlp', 10, '#96CEB4'),
            ],
            'topology': 'parallel'
        },
        5: {  # Graph Neural (Novel form)
            'name': 'Graph Form',
            'modules': [
                ModuleGene('input', 'mlp', 64, '#FF6B6B'),
                ModuleGene('graph1', 'graph', 128, '#A29BFE'),
                ModuleGene('graph2', 'graph', 128, '#74B9FF'),
                ModuleGene('aggregate', 'attention', 256, '#55EFC4'),
                ModuleGene('output', 'mlp', 10, '#96CEB4'),
            ],
            'topology': 'graph'
        }
    }
    
    form = forms[form_id]
    modules = form['modules']
    
    # Create connections based on topology
    connections = []
    if form['topology'] == 'sequential':
        for i in range(len(modules) - 1):
            connections.append(
                ConnectionGene(modules[i].id, modules[i+1].id, np.random.uniform(0.5, 1.0))
            )
    elif form['topology'] == 'recurrent':
        # Sequential + recurrent connections
        for i in range(len(modules) - 1):
            connections.append(
                ConnectionGene(modules[i].id, modules[i+1].id, np.random.uniform(0.5, 1.0))
            )
        # Add recurrent connections
        connections.append(
            ConnectionGene('recurrent1', 'recurrent1', 0.3)
        )
        connections.append(
            ConnectionGene('recurrent2', 'recurrent2', 0.3)
        )
    elif form['topology'] == 'parallel':
        # Input to both branches
        connections.append(ConnectionGene('input', 'branch1', 0.8))
        connections.append(ConnectionGene('input', 'branch2', 0.8))
        # Branches to merge
        connections.append(ConnectionGene('branch1', 'merge', 0.7))
        connections.append(ConnectionGene('branch2', 'merge', 0.7))
        # Merge to output
        connections.append(ConnectionGene('merge', 'output', 0.9))
    elif form['topology'] == 'graph':
        # Fully connected graph structure
        for i in range(len(modules) - 1):
            for j in range(i + 1, len(modules)):
                if np.random.random() > 0.3:  # 70% connection probability
                    connections.append(
                        ConnectionGene(modules[i].id, modules[j].id, np.random.uniform(0.3, 0.9))
                    )
    
    return Genotype(modules=modules, connections=connections, form_id=form_id)

def mutate(genotype: Genotype, mutation_rate: float = 0.3) -> Genotype:
    """Apply mutations to genotype"""
    mutated = genotype.copy()
    
    # Mutate module sizes
    for module in mutated.modules:
        if random.random() < mutation_rate:
            # Grow or shrink by 20%
            change = random.choice([0.8, 1.2])
            module.size = int(module.size * change)
            module.size = max(16, min(512, module.size))  # Clamp to reasonable range
    
    # Mutate connection weights
    for connection in mutated.connections:
        if random.random() < mutation_rate:
            connection.weight += np.random.normal(0, 0.1)
            connection.weight = np.clip(connection.weight, 0.1, 1.0)
    
    # Rare: Add new connection
    if random.random() < 0.1 and len(mutated.modules) > 2:
        source = random.choice(mutated.modules[:-1])
        target = random.choice(mutated.modules[1:])
        if not any(c.source == source.id and c.target == target.id for c in mutated.connections):
            mutated.connections.append(
                ConnectionGene(source.id, target.id, np.random.uniform(0.3, 0.7))
            )
    
    return mutated

def crossover(parent1: Genotype, parent2: Genotype) -> Genotype:
    """Combine two genotypes (same form only)"""
    if parent1.form_id != parent2.form_id:
        return parent1.copy()  # Can't crossover different forms
    
    child = parent1.copy()
    
    # Mix module sizes
    for i, (m1, m2) in enumerate(zip(parent1.modules, parent2.modules)):
        if random.random() < 0.5:
            child.modules[i].size = m2.size
    
    # Mix connection weights
    for i, (c1, c2) in enumerate(zip(parent1.connections, parent2.connections)):
        if random.random() < 0.5 and i < len(child.connections):
            child.connections[i].weight = c2.weight
    
    return child

def evaluate_fitness(genotype: Genotype, task_type: str) -> float:
    """
    Evaluate fitness of genotype for a given task
    
    Simulates how well this architecture would perform on the task
    """
    fitness = 0.0
    
    # Base fitness from architecture complexity
    total_params = sum(m.size for m in genotype.modules)
    complexity_score = 1.0 / (1.0 + np.exp(-total_params / 1000 + 2))  # Sigmoid
    
    # Task-specific fitness bonuses
    if task_type == 'Vision':
        # Prefer convolutional modules
        conv_count = sum(1 for m in genotype.modules if m.module_type == 'conv')
        fitness += conv_count * 0.3
        if genotype.form_id in [1, 4]:  # Sequential and Hybrid are good for vision
            fitness += 0.5
            
    elif task_type == 'Language':
        # Prefer attention modules
        attn_count = sum(1 for m in genotype.modules if m.module_type == 'attention')
        fitness += attn_count * 0.4
        if genotype.form_id == 2:  # Attention form
            fitness += 0.8
            
    elif task_type == 'Sequential':
        # Prefer recurrent modules
        rec_count = sum(1 for m in genotype.modules if m.module_type == 'recurrent')
        fitness += rec_count * 0.4
        if genotype.form_id == 3:  # Recurrent form
            fitness += 0.7
            
    elif task_type == 'Reasoning':
        # Prefer graph and attention modules
        graph_count = sum(1 for m in genotype.modules if m.module_type == 'graph')
        attn_count = sum(1 for m in genotype.modules if m.module_type == 'attention')
        fitness += (graph_count * 0.4 + attn_count * 0.3)
        if genotype.form_id in [2, 5]:  # Attention or Graph form
            fitness += 0.6
            
    elif task_type == 'Multi-Task':
        # Prefer hybrid architectures
        module_diversity = len(set(m.module_type for m in genotype.modules))
        fitness += module_diversity * 0.2
        if genotype.form_id in [4, 5]:  # Hybrid or Graph
            fitness += 0.8
    
    # Connection strength bonus
    avg_connection_weight = np.mean([c.weight for c in genotype.connections])
    fitness += avg_connection_weight * 0.5
    
    # Combine with complexity
    fitness = fitness * complexity_score + np.random.normal(0, 0.1)  # Add noise
    
    return max(0, fitness)

# ==================== VISUALIZATION ====================

def visualize_genotype(genotype: Genotype) -> go.Figure:
    """Create network graph visualization of genotype"""
    
    # Create node positions
    num_modules = len(genotype.modules)
    positions = {}
    for i, module in enumerate(genotype.modules):
        # Arrange in layers
        if genotype.form_id == 4:  # Parallel form
            if module.id in ['branch1', 'branch2']:
                x = 0.5 + (0.3 if module.id == 'branch1' else -0.3)
                y = 0.6
            elif module.id == 'input':
                x, y = 0.5, 1.0
            elif module.id == 'merge':
                x, y = 0.5, 0.3
            else:
                x, y = 0.5, 0.0
        else:
            x = i / max(num_modules - 1, 1)
            y = 0.5 + 0.3 * np.sin(i * np.pi / num_modules)
        positions[module.id] = (x, y)
    
    # Create edges
    edge_trace = []
    for conn in genotype.connections:
        if conn.source in positions and conn.target in positions:
            x0, y0 = positions[conn.source]
            x1, y1 = positions[conn.target]
            
            # Edge thickness based on weight
            width = conn.weight * 5
            
            edge_trace.append(go.Scatter(
                x=[x0, x1, None],
                y=[y0, y1, None],
                mode='lines',
                line=dict(width=width, color='rgba(125, 125, 125, 0.5)'),
                hoverinfo='text',
                text=f'Weight: {conn.weight:.2f}',
                showlegend=False
            ))
    
    # Create nodes
    node_x = []
    node_y = []
    node_color = []
    node_size = []
    node_text = []
    
    for module in genotype.modules:
        x, y = positions[module.id]
        node_x.append(x)
        node_y.append(y)
        node_color.append(module.color)
        node_size.append(module.size / 3)  # Scale for visualization
        node_text.append(f"{module.id}<br>{module.module_type}<br>size: {module.size}")
    
    node_trace = go.Scatter(
        x=node_x, y=node_y,
        mode='markers+text',
        hoverinfo='text',
        text=[m.id for m in genotype.modules],
        hovertext=node_text,
        textposition="top center",
        marker=dict(
            size=node_size,
            color=node_color,
            line=dict(width=2, color='white')
        )
    )
    
    # Create figure
    fig = go.Figure(data=edge_trace + [node_trace])
    fig.update_layout(
        showlegend=False,
        hovermode='closest',
        margin=dict(b=0,l=0,r=0,t=40),
        xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        height=400,
        title=f"Form {genotype.form_id} - Generation {genotype.generation} - Fitness: {genotype.fitness:.3f}"
    )
    
    return fig

def plot_evolution_history(history: pd.DataFrame) -> go.Figure:
    """Plot fitness evolution over time"""
    fig = px.line(history, x='generation', y='fitness', color='form',
                  title='Evolution of Fitness Across Forms',
                  labels={'fitness': 'Fitness Score', 'generation': 'Generation'})
    fig.update_traces(mode='lines+markers')
    return fig

# ==================== STREAMLIT APP ====================

def main():
    st.set_page_config(page_title="GENEVO Evolution Demo", layout="wide", page_icon="🧬")
    
    # Header
    st.title("🧬 GENEVO: True Self-Evolving Neural Architecture")
    st.markdown("""
    ### Formula: X = f(n), where n is the number of forms (1 ≤ n ≤ 5)
    
    Watch as neural architectures **evolve in real-time**, adapting their form based on environmental pressure!
    
    This demonstrates how a single evolving entity can take **multiple architectural forms**, 
    each specialized for different tasks—just like how a caterpillar transforms into a butterfly.
    """)
    
    # Sidebar controls
    st.sidebar.header("⚙️ Evolution Controls")
    
    task_type = st.sidebar.selectbox(
        "Select Task Type (Environmental Pressure)",
        ['Vision', 'Language', 'Sequential', 'Reasoning', 'Multi-Task'],
        help="The task creates selection pressure—different forms will thrive in different environments"
    )
    
    num_forms = st.sidebar.slider("Number of Forms (n)", 1, 5, 5,
                                  help="How many architectural forms can exist simultaneously")
    
    population_per_form = st.sidebar.slider("Population per Form", 2, 10, 5)
    
    mutation_rate = st.sidebar.slider("Mutation Rate", 0.1, 0.9, 0.3, 0.1)
    
    num_generations = st.sidebar.slider("Number of Generations", 5, 50, 20)
    
    # Initialize session state
    if 'history' not in st.session_state:
        st.session_state.history = []
    
    if 'current_population' not in st.session_state:
        st.session_state.current_population = None
    
    # Run evolution button
    if st.sidebar.button("🚀 Start Evolution", type="primary"):
        st.session_state.history = []
        
        # Initialize population
        population = []
        for form_id in range(1, num_forms + 1):
            for _ in range(population_per_form):
                population.append(initialize_genotype(form_id))
        
        # Progress tracking
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Evolution loop
        for gen in range(num_generations):
            status_text.text(f"Generation {gen + 1}/{num_generations}")
            
            # Evaluate fitness
            for individual in population:
                individual.fitness = evaluate_fitness(individual, task_type)
                individual.generation = gen
            
            # Record history
            for individual in population:
                st.session_state.history.append({
                    'generation': gen,
                    'form': f'Form {individual.form_id}',
                    'fitness': individual.fitness,
                    'total_params': sum(m.size for m in individual.modules)
                })
            
            # Selection: Keep top 50%
            population.sort(key=lambda x: x.fitness, reverse=True)
            survivors = population[:len(population) // 2]
            
            # Reproduction: Create offspring
            offspring = []
            while len(offspring) < len(population) - len(survivors):
                parent1 = random.choice(survivors)
                
                if random.random() < 0.7 and len(survivors) > 1:
                    # Crossover
                    parent2 = random.choice([s for s in survivors if s.form_id == parent1.form_id])
                    child = crossover(parent1, parent2)
                else:
                    # Mutation only
                    child = parent1.copy()
                
                child = mutate(child, mutation_rate)
                child.generation = gen + 1
                offspring.append(child)
            
            population = survivors + offspring
            progress_bar.progress((gen + 1) / num_generations)
        
        st.session_state.current_population = population
        status_text.text("✅ Evolution Complete!")
        st.balloons()
    
    # Display results
    if st.session_state.history:
        st.markdown("---")
        st.header("📊 Evolution Results")
        
        # Convert history to DataFrame
        history_df = pd.DataFrame(st.session_state.history)
        
        # Fitness evolution plot
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.plotly_chart(plot_evolution_history(history_df), use_container_width=True)
        
        with col2:
            st.markdown("### 📈 Final Statistics")
            final_gen = history_df[history_df['generation'] == history_df['generation'].max()]
            
            for form in sorted(final_gen['form'].unique()):
                form_data = final_gen[final_gen['form'] == form]
                avg_fitness = form_data['fitness'].mean()
                max_fitness = form_data['fitness'].max()
                
                st.metric(
                    form,
                    f"{max_fitness:.3f}",
                    f"avg: {avg_fitness:.3f}"
                )
        
        # Display best architectures
        st.markdown("---")
        st.header("🏆 Best Evolved Forms")
        
        population = st.session_state.current_population
        population.sort(key=lambda x: x.fitness, reverse=True)
        
        # Group by form and show best of each
        forms_shown = set()
        cols = st.columns(min(num_forms, 3))
        col_idx = 0
        
        for individual in population:
            if individual.form_id not in forms_shown:
                with cols[col_idx % len(cols)]:
                    st.plotly_chart(
                        visualize_genotype(individual),
                        use_container_width=True
                    )
                forms_shown.add(individual.form_id)
                col_idx += 1
                
                if len(forms_shown) >= num_forms:
                    break
        
        # Analysis
        st.markdown("---")
        st.header("🔬 Evolutionary Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Form Distribution")
            form_counts = final_gen['form'].value_counts()
            fig = px.bar(x=form_counts.index, y=form_counts.values,
                        labels={'x': 'Form', 'y': 'Count'},
                        title='Population Distribution by Form')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("### Parameter Evolution")
            fig = px.line(history_df.groupby(['generation', 'form'])['total_params'].mean().reset_index(),
                         x='generation', y='total_params', color='form',
                         title='Average Parameters Over Time')
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Instructions
        st.info("👈 Configure evolution parameters in the sidebar and click '🚀 Start Evolution' to begin!")
        
        st.markdown("""
        ### What You'll See:
        
        1. **Multiple Forms**: Each represents a different architectural paradigm (Sequential, Attention, Recurrent, Hybrid, Graph)
        2. **Natural Selection**: Forms better suited to the task will thrive
        3. **Adaptation**: Architectures will grow, shrink, and rewire based on pressure
        4. **Emergence**: Watch how the population shifts toward optimal forms
        
        ### The Five Forms:
        
        - **Form 1 (Sequential)**: Traditional CNN-like architecture
        - **Form 2 (Attention)**: Transformer-like with attention mechanisms
        - **Form 3 (Recurrent)**: RNN-like with temporal dependencies
        - **Form 4 (Hybrid Parallel)**: ResNet-like with parallel pathways
        - **Form 5 (Graph)**: Graph Neural Network with flexible connectivity
        
        Each form has its strengths! The environment (task type) determines which survives. 🌱→🌳
        """)

if __name__ == "__main__":
    main()
