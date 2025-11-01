# Morphogenetic Neural Architectures: A Universal Genotype-Phenotype Framework for Indefinite Architectural Evolution

**A Comprehensive Theoretical and Empirical Investigation into Evolvable Meta-Architectures as the Foundation for Artificial General Intelligence**

---

## Abstract

We present a comprehensive theoretical and computational framework for neural architectures as evolutionary genetic programs capable of indefinite morphological variation through mutation, recombination, and selection pressure. Rather than selecting among discrete architectural paradigms (Transformers, Mixture of Experts, Recurrent Networks, Graph Networks), we propose encoding architecture itself as evolvable genotypes subject to Darwinian dynamics—yielding a meta-architecture that subsumes all fixed topologies as degenerate cases within an infinite-dimensional morphospace. This fundamentally dissolves the architectural selection problem by treating structure as a dynamically optimizable phenotype emerging from compact genetic encodings through developmental processes. We formalize this through differentiable genotype-phenotype mappings, compositional genetic operators, and population-based meta-optimization across task manifolds. Our framework integrates three nested timescales of adaptation: evolutionary search over architectural space (generations), developmental growth from genetic blueprints (ontogeny), and intra-lifetime learning through gradient descent and local plasticity rules (experience). We demonstrate theoretically that this approach achieves universal expressivity over the space of computable architectures, while empirically showing superior performance on compositional reasoning, continual learning, and few-shot adaptation compared to fixed-topology baselines. This work establishes evolutionary neural architectures as a viable path toward artificial general intelligence by removing the fundamental constraint of architectural stasis.

**Keywords**: Neuroevolution, Genetic Algorithms, Neural Architecture Search, Meta-Learning, Developmental Encoding, Morphogenetic Computation, Evolutionary Computation, AGI

---

## Table of Contents

1. Introduction and Motivation
2. Theoretical Foundations
3. The Genotype-Phenotype Framework
4. Genetic Operators and Evolutionary Dynamics
5. Developmental Mapping and Morphogenesis
6. Multi-Scale Learning Mechanisms
7. Population-Based Meta-Optimization
8. Computational Implementation
9. Theoretical Analysis and Convergence Properties
10. Empirical Validation
11. Applications to AGI Challenges
12. Philosophical and Epistemological Implications
13. Open Problems and Future Directions
14. Conclusion

---

*[Sections 1-3.3 from original document]*

### 3.4 Genotype Complexity and Kolmogorov Compression

A critical advantage of the genotype-phenotype framework is **developmental compression**: complex phenotypes emerge from compact genetic specifications through recursive growth programs.

**Definition 3.1** (Genetic Complexity): The genetic complexity κ(n) of phenotype n is the length of the shortest genotype that develops into n:

κ(n) = min{|g| : Φ(g) ≈_ε n}

where |g| denotes the description length of genotype g and ≈_ε denotes ε-functional equivalence.

**Theorem 3.1** (Developmental Compression): *For regular, highly structured networks with L layers and D-dimensional representations:*

κ(n) = O(log(L) + log(D))

*compared to direct specification requiring:*

|n|_direct = O(L · D²)

**Proof**: A genotype can specify "stack L identical layers" using logarithmic description length through:
```python
g = ModuleGene(type=ATTENTION, d_model=D) + 
    DevelopmentalParams(base_layers=L, repetition=True)
```

The repetition instruction requires only log(L) bits to encode L. Module hyperparameters require log(D) bits. Total genetic complexity is thus O(log(L·D)) compared to O(L·D²) parameters in the realized network. ∎

**Corollary 3.1**: Evolution can search exponentially larger phenotype spaces by operating on compact genotypes. The effective search space size is |G| rather than |N|, where |G| << |N|.

This explains biological efficiency: The human genome (3×10⁹ base pairs ≈ 750 MB) specifies development of 86 billion neurons with 10¹⁴ synapses—a compression ratio exceeding 10⁸.

### 3.5 Functional Modularity and Compositionality

**Definition 3.2** (Functional Module): A subset M' ⊂ M of module genes constitutes a functional module if:
1. **Cohesion**: Modules in M' are highly interconnected (dense subgraph in C)
2. **Separation**: Connections between M' and M \ M' are sparse (module boundary)
3. **Functional coherence**: M' computes a semantically meaningful sub-function

**Theorem 3.2** (Compositional Generalization): *Networks with modular genotype structure generalize systematically to novel combinations of learned sub-functions.*

**Proof Sketch**: 
1. Each functional module f_i implements a reusable computation (e.g., "compare two inputs", "track temporal context")
2. Novel tasks may require new compositions of existing modules
3. Modular networks can recombine f_1, f_2, ... , f_k into f_new = h(f_i, f_j, ...) without relearning primitives
4. Non-modular networks entangle representations, preventing clean composition
∎

**Example**: Consider learning arithmetic:
- Module M₁: Number encoder (visual → abstract number)
- Module M₂: Addition operator
- Module M₃: Multiplication operator

A modular genotype allows these to be composed: M₁ → M₂ for addition tasks, M₁ → M₃ for multiplication tasks, M₁ → M₃ → M₂ for "multiply then add". Non-modular networks must learn each combination from scratch.

**Modularity emergence**: Evolution naturally discovers modular structures because:
1. **Reusability**: Modules amortize learning cost across tasks
2. **Evolvability**: Modular genotypes have smoother fitness landscapes (local changes don't disrupt entire system)
3. **Robustness**: Failure of one module doesn't cascade throughout network

---

## 4. Genetic Operators and Evolutionary Dynamics

Evolution operates through variation (mutation, recombination) and selection. We now formalize these genetic operators.

### 4.1 Mutation Operators

Mutations introduce heritable variation. We define multiple mutation types operating at different structural levels:

#### 4.1.1 Parametric Mutations

**Continuous parameter perturbation**:

```python
def mutate_hyperparameters(genotype: Genotype, 
                           mutation_rate: float = 0.1,
                           mutation_strength: float = 0.2) -> Genotype:
    """
    Gaussian mutations on continuous hyperparameters
    
    Each hyperparameter mutates independently with probability mutation_rate,
    applying Gaussian noise with std = mutation_strength * current_value
    """
    mutated = copy.deepcopy(genotype)
    
    for module in mutated.modules:
        for param_name, param_value in module.hyperparams.items():
            if random.random() < mutation_rate:
                if isinstance(param_value, float):
                    # Continuous: Gaussian perturbation
                    noise = np.random.normal(0, mutation_strength * abs(param_value))
                    module.hyperparams[param_name] = param_value + noise
                    
                elif isinstance(param_value, int):
                    # Discrete: Poisson perturbation
                    delta = np.random.poisson(mutation_strength * param_value)
                    if random.random() < 0.5:
                        delta = -delta
                    module.hyperparams[param_name] = max(1, param_value + delta)
    
    return mutated
```

**Theorem 4.1** (Parametric Continuity): *Gaussian mutations on hyperparameters induce continuous trajectories in phenotype space, enabling gradient-like evolutionary search.*

**Proof**: The phenotype mapping Φ is continuous in hyperparameters (small changes in d_model, n_heads, etc. produce small functional changes). Gaussian mutations sample from local neighborhoods in parameter space, inducing local search in phenotype space. ∎

#### 4.1.2 Topological Mutations

**Adding connections**:

```python
def add_connection_mutation(genotype: Genotype, 
                           innovation_rate: float = 0.05) -> Genotype:
    """
    Create new connection between existing modules
    
    Samples uniformly from pairs of modules without direct connection,
    adds edge with random weight initialization
    """
    mutated = copy.deepcopy(genotype)
    
    if random.random() < innovation_rate:
        # Find all possible new connections
        existing_edges = {(c.source, c.target) for c in genotype.connections}
        possible_new = []
        
        for source in genotype.modules:
            for target in genotype.modules:
                if source.id != target.id and (source.id, target.id) not in existing_edges:
                    # Check for cycles if acyclic constraint
                    if not would_create_cycle(mutated, source.id, target.id):
                        possible_new.append((source.id, target.id))
        
        if possible_new:
            source, target = random.choice(possible_new)
            new_connection = ConnectionGene(
                source=source,
                target=target,
                weight=np.random.normal(0, 0.1),
                properties=ConnectionProperties()
            )
            mutated.connections.append(new_connection)
    
    return mutated
```

**Removing connections** (pruning):

```python
def remove_connection_mutation(genotype: Genotype,
                               pruning_rate: float = 0.02) -> Genotype:
    """Remove weak or redundant connections"""
    mutated = copy.deepcopy(genotype)
    
    # Don't prune below minimum connectivity
    if len(mutated.connections) > len(mutated.modules):
        for connection in list(mutated.connections):
            if random.random() < pruning_rate:
                # Ensure network stays connected
                if is_still_connected(mutated, connection):
                    mutated.connections.remove(connection)
    
    return mutated
```

**Adding modules** (neurogenesis):

```python
def add_module_mutation(genotype: Genotype,
                       module_library: List[ModuleType],
                       addition_rate: float = 0.03) -> Genotype:
    """Insert new computational module"""
    mutated = copy.deepcopy(genotype)
    
    if random.random() < addition_rate:
        # Sample module type from library (with bias toward successful types)
        new_type = sample_module_type(module_library, fitness_bias=True)
        
        # Generate random but reasonable hyperparameters
        hyperparams = initialize_hyperparameters(
            new_type, 
            reference_genotype=genotype
        )
        
        new_module = ModuleGene(
            id=generate_unique_id(),
            type=new_type,
            hyperparams=hyperparams,
            activation=sample_activation(),
            normalization=sample_normalization(),
            initialization="xavier_uniform"
        )
        
        mutated.modules.append(new_module)
        
        # Connect to existing modules (random or learned policy)
        add_connections_to_new_module(mutated, new_module)
    
    return mutated
```

**Removing modules** (apoptosis):

```python
def remove_module_mutation(genotype: Genotype,
                          removal_rate: float = 0.02) -> Genotype:
    """Remove redundant or low-utility modules"""
    mutated = copy.deepcopy(genotype)
    
    # Don't reduce below minimum architecture
    if len(mutated.modules) > 3:  # input, processing, output minimum
        for module in list(mutated.modules):
            if random.random() < removal_rate:
                if not is_essential_module(module, mutated):
                    # Remove module and its connections
                    mutated.modules.remove(module)
                    mutated.connections = [c for c in mutated.connections 
                                          if c.source != module.id and c.target != module.id]
    
    return mutated
```

#### 4.1.3 Plasticity Rule Mutations

```python
def mutate_plasticity(genotype: Genotype,
                     plasticity_mutation_rate: float = 0.1) -> Genotype:
    """Modify learning rules and their metaparameters"""
    mutated = copy.deepcopy(genotype)
    
    for plasticity_rule in mutated.plasticity_rules:
        if random.random() < plasticity_mutation_rate:
            # Option 1: Mutate metaparameters
            if random.random() < 0.7:
                for param_name in plasticity_rule.metaparams:
                    plasticity_rule.metaparams[param_name] *= np.random.lognormal(0, 0.3)
            
            # Option 2: Change plasticity type
            else:
                plasticity_rule.rule_type = sample_plasticity_type()
                plasticity_rule.metaparams = initialize_metaparams(plasticity_rule.rule_type)
    
    # Sometimes add new plasticity rule
    if random.random() < 0.05:
        new_rule = PlasticityRule(
            target=random.choice([m.id for m in mutated.modules]),
            rule_type=sample_plasticity_type(),
            metaparams=initialize_metaparams(),
            modulation=None
        )
        mutated.plasticity_rules.append(new_rule)
    
    return mutated
```

#### 4.1.4 Developmental Mutations

```python
def mutate_developmental_params(genotype: Genotype,
                               dev_mutation_rate: float = 0.1) -> Genotype:
    """Modify morphogenetic parameters"""
    mutated = copy.deepcopy(genotype)
    dev = mutated.developmental_params
    
    if random.random() < dev_mutation_rate:
        # Mutate scaling parameters
        dev.base_layers = max(1, dev.base_layers + np.random.randint(-2, 3))
        dev.width_multiplier *= np.random.lognormal(0, 0.2)
        
        # Mutate pruning schedule
        if random.random() < 0.3:
            dev.initial_connection_density += np.random.normal(0, 0.1)
            dev.initial_connection_density = np.clip(dev.initial_connection_density, 0.1, 1.0)
        
        # Toggle features
        if random.random() < 0.1:
            dev.neurogenesis = not dev.neurogenesis
        if random.random() < 0.1:
            dev.task_conditional = not dev.task_conditional
    
    return mutated
```

### 4.2 Recombination Operators

Recombination (crossover) combines genetic material from two parents, enabling composition of successful building blocks.

#### 4.2.1 Module-Level Crossover

**Uniform module crossover**:

```python
def uniform_module_crossover(parent1: Genotype, 
                            parent2: Genotype,
                            crossover_rate: float = 0.5) -> Tuple[Genotype, Genotype]:
    """
    Exchange modules between parents at random
    
    Each module inherited from parent1 or parent2 independently.
    Connections adjusted to maintain validity.
    """
    child1 = Genotype(modules=[], connections=[], plasticity_rules=[], developmental_params=None)
    child2 = Genotype(modules=[], connections=[], plasticity_rules=[], developmental_params=None)
    
    # Inherit modules
    for m1, m2 in zip(parent1.modules, parent2.modules):
        if random.random() < crossover_rate:
            child1.modules.append(copy.deepcopy(m1))
            child2.modules.append(copy.deepcopy(m2))
        else:
            child1.modules.append(copy.deepcopy(m2))
            child2.modules.append(copy.deepcopy(m1))
    
    # Inherit connections (only those between inherited modules)
    child1.connections = inherit_valid_connections(parent1, parent2, child1.modules, crossover_rate)
    child2.connections = inherit_valid_connections(parent2, parent1, child2.modules, crossover_rate)
    
    # Inherit plasticity rules
    child1.plasticity_rules = inherit_plasticity(parent1, parent2, crossover_rate)
    child2.plasticity_rules = inherit_plasticity(parent2, parent1, crossover_rate)
    
    # Inherit developmental parameters
    child1.developmental_params = blend_dev_params(parent1.developmental_params, 
                                                    parent2.developmental_params)
    child2.developmental_params = blend_dev_params(parent2.developmental_params, 
                                                    parent1.developmental_params)
    
    return child1, child2
```

#### 4.2.2 Functional Module Preservation

Key innovation: **Respect functional modularity during crossover** to avoid breaking coherent computational units.

```python
def modular_crossover(parent1: Genotype,
                     parent2: Genotype) -> Tuple[Genotype, Genotype]:
    """
    Crossover that preserves functional modules
    
    1. Identify functional modules in each parent
    2. Exchange entire modules rather than individual genes
    3. Repair connections at module boundaries
    """
    # Detect functional modules via graph clustering
    modules1 = detect_functional_modules(parent1)
    modules2 = detect_functional_modules(parent2)
    
    # Match similar modules between parents (by function/structure)
    module_correspondence = match_modules(modules1, modules2)
    
    # Create children by swapping matched modules
    child1 = copy.deepcopy(parent1)
    child2 = copy.deepcopy(parent2)
    
    for (mod1, mod2) in module_correspondence:
        if random.random() < 0.5:
            # Swap this functional module pair
            swap_functional_module(child1, child2, mod1, mod2)
            
            # Repair boundary connections
            repair_module_boundaries(child1, mod2)
            repair_module_boundaries(child2, mod1)
    
    return child1, child2


def detect_functional_modules(genotype: Genotype) -> List[FunctionalModule]:
    """
    Identify cohesive subgraphs in connectivity graph
    
    Uses community detection (Louvain algorithm) on module connectivity
    """
    # Build adjacency matrix from connections
    adjacency = build_adjacency_matrix(genotype)
    
    # Apply graph clustering
    communities = louvain_clustering(adjacency)
    
    # Each community is a functional module
    functional_modules = []
    for community in communities:
        module_genes = [m for m in genotype.modules if m.id in community]
        internal_connections = [c for c in genotype.connections 
                               if c.source in community and c.target in community]
        
        functional_modules.append(FunctionalModule(
            genes=module_genes,
            connections=internal_connections,
            boundary=compute_boundary(community, genotype)
        ))
    
    return functional_modules
```

**Theorem 4.2** (Modular Recombination Theorem): *Modular crossover preserves fitness with probability p > 1 - ε if functional modules are properly identified, whereas random crossover has p ≈ 0.5 for complex genotypes.*

**Proof**: 
1. Random crossover breaks dependencies with probability proportional to coupling strength
2. If module M_i depends on internal state across genes {g_1, ..., g_k}, random crossover splits these with prob ≈ 1 - 2^(-k)
3. Modular crossover keeps {g_1, ..., g_k} together, preserving M_i's function
4. Thus modular crossover maintains fitness except at module boundaries (sparse by definition)
∎

#### 4.2.3 Hyperparameter Blending

For continuous hyperparameters, use arithmetic recombination:

```python
def blend_hyperparameters(parent1: ModuleGene,
                         parent2: ModuleGene,
                         alpha: float = 0.5) -> Dict:
    """
    Blend continuous hyperparameters between parents
    
    Args:
        alpha: Blending coefficient (0 = parent1, 1 = parent2)
    """
    blended = {}
    for param_name in parent1.hyperparams:
        if isinstance(parent1.hyperparams[param_name], float):
            v1 = parent1.hyperparams[param_name]
            v2 = parent2.hyperparams[param_name]
            blended[param_name] = (1 - alpha) * v1 + alpha * v2
        elif isinstance(parent1.hyperparams[param_name], int):
            v1 = parent1.hyperparams[param_name]
            v2 = parent2.hyperparams[param_name]
            blended[param_name] = int(round((1 - alpha) * v1 + alpha * v2))
        else:
            # Discrete: randomly choose
            blended[param_name] = random.choice([parent1.hyperparams[param_name],
                                                 parent2.hyperparams[param_name]])
    return blended
```

### 4.3 Selection Mechanisms

Selection determines which genotypes reproduce. We employ multiple selection strategies operating simultaneously:

#### 4.3.1 Multi-Objective Pareto Selection

```python
def pareto_selection(population: List[Genotype],
                    fitnesses: np.ndarray,  # Shape: [pop_size, num_objectives]
                    selection_size: int) -> List[Genotype]:
    """
    Select from Pareto front for multi-objective optimization
    
    Preserves diverse solutions optimizing different objective trade-offs
    """
    # Compute Pareto ranks
    ranks = fast_non_dominated_sort(fitnesses)
    
    # Compute crowding distances (diversity within ranks)
    distances = crowding_distance(fitnesses, ranks)
    
    # Select by rank, breaking ties by crowding distance
    selected = []
    for rank in range(max(ranks) + 1):
        rank_individuals = [(i, distances[i]) for i in range(len(population)) 
                           if ranks[i] == rank]
        rank_individuals.sort(key=lambda x: x[1], reverse=True)  # Higher distance better
        
        for idx, _ in rank_individuals:
            selected.append(population[idx])
            if len(selected) >= selection_size:
                return selected
    
    return selected


def fast_non_dominated_sort(fitnesses: np.ndarray) -> np.ndarray:
    """
    Deb's fast non-dominated sorting algorithm
    
    Returns array of Pareto ranks (0 = non-dominated front)
    """
    pop_size, num_obj = fitnesses.shape
    ranks = np.zeros(pop_size, dtype=int)
    
    # For each individual, compute domination relationships
    domination_counts = np.zeros(pop_size)
    dominated_solutions = [[] for _ in range(pop_size)]
    
    for i in range(pop_size):
        for j in range(i + 1, pop_size):
            if dominates(fitnesses[i], fitnesses[j]):
                dominated_solutions[i].append(j)
                domination_counts[j] += 1
            elif dominates(fitnesses[j], fitnesses[i]):
                dominated_solutions[j].append(i)
                domination_counts[i] += 1
    
    # Assign ranks via breadth-first search
    current_rank = 0
    front = np.where(domination_counts == 0)[0].tolist()
    
    while front:
        next_front = []
        for i in front:
            ranks[i] = current_rank
            for j in dominated_solutions[i]:
                domination_counts[j] -= 1
                if domination_counts[j] == 0:
                    next_front.append(j)
        front = next_front
        current_rank += 1
    
    return ranks


def dominates(f1: np.ndarray, f2: np.ndarray) -> bool:
    """Check if fitness vector f1 dominates f2"""
    return np.all(f1 >= f2) and np.any(f1 > f2)
```

#### 4.3.2 Tournament Selection

For single-objective or aggregate fitness:

```python
def tournament_selection(population: List[Genotype],
                        fitnesses: np.ndarray,
                        tournament_size: int = 4) -> Genotype:
    """
    Select via tournament: sample k individuals, return fittest
    
    Creates selection pressure while maintaining diversity
    """
    tournament_indices = np.random.choice(len(population), 
                                         size=tournament_size, 
                                         replace=False)
    tournament_fitnesses = fitnesses[tournament_indices]
    winner_idx = tournament_indices[np.argmax(tournament_fitnesses)]
    return copy.deepcopy(population[winner_idx])
```

#### 4.3.3 Fitness Sharing (Speciation)

Prevent premature convergence by maintaining diversity:

```python
def fitness_sharing(population: List[Genotype],
                   fitnesses: np.ndarray,
                   sigma_share: float = 3.0) -> np.ndarray:
    """
    Apply fitness sharing to maintain population diversity
    
    Reduces fitness of individuals in crowded regions of genotype space
    """
    shared_fitnesses = fitnesses.copy()
    
    for i in range(len(population)):
        niche_count = 0.0
        for j in range(len(population)):
            distance = genotype_distance(population[i], population[j])
            if distance < sigma_share:
                niche_count += 1 - (distance / sigma_share)
        
        shared_fitnesses[i] /= niche_count
    
    return shared_fitnesses


def genotype_distance(g1: Genotype, g2: Genotype) -> float:
    """
    Compute structural distance between genotypes
    
    Combines:
    - Topological distance (graph edit distance)
    - Parametric distance (hyperparameter L2 norm)
    - Functional distance (output correlation on test inputs)
    """
    # Topological component
    graph_distance = graph_edit_distance(g1.connections, g2.connections)
    
    # Parametric component
    param_distance = 0.0
    for m1, m2 in zip(g1.modules, g2.modules):
        for param_name in m1.hyperparams:
            if param_name in m2.hyperparams:
                param_distance += (m1.hyperparams[param_name] - 
                                  m2.hyperparams[param_name])**2
    param_distance = np.sqrt(param_distance)
    
    # Functional component (requires phenotype evaluation)
    functional_distance = behavioral_distance(g1, g2)
    
    # Weighted combination
    total_distance = (0.3 * graph_distance + 
                     0.3 * param_distance + 
                     0.4 * functional_distance)
    
    return total_distance
```

#### 4.3.4 Novelty Search

Complement fitness-based selection with novelty rewards:

```python
def novelty_search(population: List[Genotype],
                  archive: List[Genotype],
                  k_nearest: int = 15) -> np.ndarray:
    """
    Compute novelty scores based on behavioral uniqueness
    
    Rewards exploration of unvisited regions in behavior space
    """
    novelty_scores = np.zeros(len(population))
    
    for i, genotype in enumerate(population):
        # Extract behavioral characterization
        behavior = extract_behavior(genotype)
        
        # Compute distances to k nearest neighbors (in population + archive)
        all_behaviors = [extract_behavior(g) for g in population + archive]
        distances = [behavioral_distance(behavior, b) for b in all_behaviors]
        distances.sort()
        
        # Novelty = average distance to k nearest neighbors
        novelty_scores[i] = np.mean(distances[1:k_nearest+1])  # Exclude self (distance 0)
    
    return novelty_scores


def extract_behavior(genotype: Genotype) -> np.ndarray:
    """
    Extract behavioral characterization for novelty computation
    
    Examples:
    - Final layer activations on diverse inputs
    - Trajectory through state space during task execution
    - Attention pattern statistics
    """
    phenotype = develop_phenotype(genotype)
    
    # Run on diverse probe inputs
    probe_inputs = generate_probe_set()
    activations = []
    
    for x in probe_inputs:
        output = phenotype.forward(x)
        activations.append(output.detach().numpy())
    
    # Characterization = statistics over activations
    behavior = np.concatenate([
        np.mean(activations, axis=0),
        np.std(activations, axis=0),
        np.percentile(activations, [25, 75], axis=0).flatten()
    ])
    
    return behavior
```

### 4.4 Combined Selection Strategy

In practice, we use weighted combination:

```python
def select_parents(population: List[Genotype],
                  fitnesses_multi: np.ndarray,
                  archive: List[Genotype],
                  selection_size: int) -> List[Genotype]:
    """
    Hybrid selection combining multiple objectives and novelty
    """
    # Multi-objective Pareto selection (70% of parents)
    pareto_parents = pareto_selection(
        population, 
        fitnesses_multi, 
        int(0.7 * selection_size)
    )
    
    # Novelty-based selection (20% of parents)
    novelty_scores = novelty_search(population, archive)
    novelty_parents = [population[i] for i in np.argsort(novelty_scores)[-int(0.2 * selection_size):]]
    
    # Tournament selection on aggregate fitness (10% of parents)
    aggregate_fitness = np.mean(fitnesses_multi, axis=1)
    tournament_parents = [tournament_selection(population, aggregate_fitness) 
                         for _ in range(int(0.1 * selection_size))]
    
    all_parents = pareto_parents + novelty_parents + tournament_parents
    return all_parents
```

### 4.5 Evolutionary Algorithm

Complete evolutionary loop:

```python
def evolve_architectures(
    initial_population: List[Genotype],
    task_distribution: TaskDistribution,
    num_generations: int = 1000,
    population_size: int = 200,
    offspring_size: int = 400,
    mutation_rate: float = 0.3,
    crossover_rate: float = 0.7
) -> List[Genotype]:
    """
    Main evolutionary algorithm for architecture search
    
    Returns final population of evolved genotypes
    """
    population = initial_population
    archive = []  # Archive for novelty search
    
    for generation in range(num_generations):
        print(f"Generation {generation}/{num_generations}")
        
        # Step 1: Evaluate fitness
        fitnesses = evaluate_population(population, task_distribution)
        
        # Step 2: Update archive (add novel high-fitness individuals)
        update_archive(archive, population, fitnesses)
        
        # Step 3: Select parents
        parents = select_parents(
            population, 
            fitnesses,
            archive,
            selection_size=population_size
        )
        
        # Step 4: Generate offspring
        offspring = []
        while len(offspring) < offspring_size:
            # Crossover
            if random.random() < crossover_rate and len(parents) >= 2:
                parent1, parent2 = random.sample(parents, 2)
                child1, child2 = modular_crossover(parent1, parent2)
                offspring.extend([child1, child2])
            else:
                # Asexual reproduction
                parent = random.choice(parents)
                offspring.append(copy.deepcopy(parent))
        
        # Step 5: Apply mutations
        for i in range(len(offspring)):
            if random.random() < mutation_rate:
                # Apply multiple mutation types
                offspring[i] = mutate_hyperparameters(offspring[i])
                offspring[i] = add_connection_mutation(offspring[i])
                offspring[i] = remove_connection_mutation(offspring[i])
                offspring[i] = add_module_mutation(offspring[i])
                offspring[i] = remove_module_mutation(offspring[i])
                offspring[i] = mutate_plasticity(offspring[i])
                offspring[i] = mutate_developmental_params(offspring[i])
        
        # Step 6: Evaluate offspring
        offspring_fitnesses = evaluate_population(offspring, task_distribution)
        
        # Step 7: Environmental selection (survival of fittest)
        combined_population = population + offspring
        combined_fitnesses = np.vstack([fitnesses, offspring_fitnesses])
        
        # Select next generation
        population = environmental_selection(
            combined_population,
            combined_fitnesses,
            population_size
        )
        
        # Step 8: Log statistics
        log_generation_stats(generation, population, fitnesses)
        
        # Step 9: Checkpointing
        if generation % 50 == 0:
            save_checkpoint(generation, population, archive)
    
    return population
```

---

## 5. Developmental Mapping and Morphogenesis

The genotype-to-phenotype mapping Φ: G → N is not merely parameter instantiation but a **developmental process** analogous to biological morphogenesis. This section formalizes developmental computation.

### 5.1 Computational Embryology

**Definition 5.1** (Developmental Process): A developmental process is a sequence of transformations D = {d_t : S_t → S_{t+1}} where:
- S_t represents the structural state at developmental time t
- S_0 = g (genotype)
- S_T = n (mature phenotype)
- Each d_t interprets genetic instructions to modify structure

The complete developmental mapping is the composition:

Φ = d_T ∘ d_{T-1} ∘ ... ∘ d_1 ∘ d_0

#### 5.1.1 Staged Development

We implement development as discrete stages:

```python
def develop_phenotype_staged(genotype: Genotype,
                             context: Optional[Context] = None) -> Phenotype:
    """
    Multi-stage morphogenesis from genotype to phenotype
    
    Stages:
    1. Specification: Determine basic structure from genetic blueprint
    2. Proliferation: Grow modules according to developmental parameters
    3. Differentiation: Specialize modules based on position/context
    4. Synaptogenesis: Establish connections between modules
    5. Pruning: Remove weak or redundant connections
    6. Maturation: Finalize parameters and learning rules
    """
    
    # Stage 1: Specification
    structure = specify_basic_structure(genotype)
    
    # Stage 2: Proliferation
    structure = proliferate_modules(structure, genotype.developmental_params)
    
    # Stage 3: Differentiation
    structure = differentiate_modules(structure, context)
    
    # Stage 4: Synaptogenesis
    structure = establish_connections(structure, genotype.connections)
    
    # Stage 5: Pruning
    structure = developmental_pruning(structure, genotype.developmental_params)
    
    # Stage 6: Maturation
    phenotype = finalize_phenotype(structure, genotype.plasticity_rules)
    
    return phenotype
```

**Stage 1: Specification**

```python
def specify_basic_structure(genotype: Genotype) -> DevelopmentalStructure:
    """
    Create initial scaffold from module genes
    
    Each module gene becomes a proto-module with placeholder parameters
    """
    structure = DevelopmentalStructure()
    
    for module_gene in genotype.modules:
        proto_module = ProtoModule(
            id=module_gene.id,
            type=module_gene.type,
            base_hyperparams=module_gene.hyperparams,
            position=None,  # To be determined by proliferation
            differentiation_state="undifferentiated"
        )
        structure.add_proto_module(proto_module)
    
    return structure
```

**Stage 2: Proliferation**

```python
def proliferate_modules(structure: DevelopmentalStructure,
                       dev_params: DevelopmentalParams) -> DevelopmentalStructure:
    """
    Expand structure through repetition and scaling
    
    Implements patterns like:
    - "Stack this module N times" (layer repetition)
    - "Create a pyramid with widths [512, 256, 128]" (hierarchical scaling)
    - "Duplicate module with parameter sharing" (tied weights)
    """
    expanded_structure = DevelopmentalStructure()
    
    for proto_module in structure.proto_modules:
        # Determine proliferation count
        if dev_params.base_layers > 1:
            # Replicate this module
            for layer_idx in range(dev_params.base_layers):
                replicated_module = copy.deepcopy(proto_module)
                replicated_module.id = f"{proto_module.id}_L{layer_idx}"
                replicated_module.position = layer_idx
                
                # Apply depth-dependent scaling
                scale_factor = dev_params.depth_schedule(layer_idx)
                replicated_module.scale_hyperparams(scale_factor)
                
                expanded_structure.add_proto_module(replicated_module)
        else:
            # Single instance
            expanded_structure.add_proto_module(proto_module)
    
    # Apply global width multiplier
    for module in expanded_structure.proto_modules:
        module.base_hyperparams['d_model'] = int(
            module.base_hyperparams['d_model'] * dev_params.width_multiplier
        )
    
    return expanded_structure
```

**Stage 3: Differentiation**

```python
def differentiate_modules(structure: DevelopmentalStructure,
                         context: Optional[Context]) -> DevelopmentalStructure:
    """
    Specialize modules based on positional and contextual information
    
    Biological analogy: HOX genes specify anterior-posterior identity
    Here: Position in network determines specialization
    """
    for module in structure.proto_modules:
        # Position-dependent differentiation
        if module.position is not None:
            if module.position == 0:
                # First layer: input processing specialization
                module.specialize_for_input()
            elif module.position == structure.depth() - 1:
                # Last layer: output generation specialization
                module.specialize_for_output()
            else:
                # Middle layers: feature transformation
                module.specialize_for_processing()
        
        # Context-dependent differentiation
        if context is not None:
            # Adapt to task characteristics
            if context.task_type == "vision":
                module.add_spatial_inductive_bias()
            elif context.task_type == "language":
                module.add_sequential_inductive_bias()
            elif context.task_type == "reasoning":
                module.add_compositional_structure()
        
        # Mark as differentiated
        module.differentiation_state = "differentiated"
    
    return structure
```

**Stage 4: Synaptogenesis**

```python
def establish_connections(structure: DevelopmentalStructure,
                         connection_genes: List[ConnectionGene]) -> DevelopmentalStructure:
    """
    Create synaptic connections between differentiated modules
    
    Interprets connection genes to build computational graph
    """
    for conn_gene in connection_genes:
        # Find source and target modules (may have been proliferated)
        source_modules = structure.find_modules_by_base_id(conn_gene.source)
        target_modules = structure.find_modules_by_base_id(conn_gene.target)
        
        # Connection pattern depends on proliferation
        if len(source_modules) == len(target_modules):
            # One-to-one connections (parallel pathways)
            for src, tgt in zip(source_modules, target_modules):
                structure.add_connection(src.id, tgt.id, conn_gene.properties)
        
        elif len(source_modules) == 1:
            # One-to-many (broadcast)
            for tgt in target_modules:
                structure.add_connection(source_modules[0].id, tgt.id, conn_gene.properties)
        
        elif len(target_modules) == 1:
            # Many-to-one (aggregation)
            for src in source_modules:
                structure.add_connection(src.id, target_modules[0].id, conn_gene.properties)
        
        else:
            # Many-to-many (full connectivity or learned pattern)
            pattern = determine_connectivity_pattern(source_modules, target_modules)
            for src_idx, tgt_idx in pattern:
                structure.add_connection(
                    source_modules[src_idx].id,
                    target_modules[tgt_idx].id,
                    conn_gene.properties
                )
    
    # Add implicit connections (residual, layer normalization placement)
    structure.add_implicit_connections()
    
    return structure
```

**Stage 5: Developmental Pruning**

```python
def developmental_pruning(structure: DevelopmentalStructure,
                         dev_params: DevelopmentalParams) -> DevelopmentalStructure:
    """
    Remove connections based on developmental rules
    
    Analogous to synaptic pruning in neurodevelopment
    """
    if dev_params.initial_connection_density < 1.0:
        # Stochastic pruning
        for connection in list(structure.connections):
            if random.random() > dev_params.initial_connection_density:
                structure.remove_connection(connection)
    
    # Activity-independent pruning (before any learning)
    # Remove topologically redundant connections
    redundant = identify_redundant_connections(structure)
    for connection in redundant:
        structure.remove_connection(connection)
    
    # Ensure connectivity
    if not structure.is_connected():
        structure.restore_minimum_connectivity()
    
    return structure
```

**Stage 6: Maturation**

```python
def finalize_phenotype(structure: DevelopmentalStructure,
                      plasticity_rules: List[PlasticityRule]) -> Phenotype:
    """
    Convert developmental structure to executable phenotype
    
    Instantiate actual PyTorch modules and set up learning mechanisms
    """
    # Instantiate neural modules
    modules = {}
    for proto_module in structure.proto_modules:
        module_instance = instantiate_module(
            proto_module.type,
            proto_module.base_hyperparams
        )
        initialize_weights(module_instance, proto_module.initialization)
        modules[proto_module.id] = module_instance
    
    # Build computation graph
    graph = ComputationGraph()
    for conn in structure.connections:
        edge = create_edge(
            modules[conn.source],
            modules[conn.target],
            conn.properties
        )
        graph.add_edge(edge)
    
    # Setup plasticity mechanisms
    plasticity = {}
    for rule in plasticity_rules:
        mechanism = instantiate_plasticity_mechanism(rule)
        plasticity[rule.target] = mechanism
    
    # Create phenotype
    phenotype = Phenotype(
        modules=modules,
        graph=graph,
        plasticity=plasticity,
        parameters=collect_parameters(modules)
    )
    
    # Ensure differentiability
    phenotype = ensure_gradient_flow(phenotype)
    
    return phenotype
```

### 5.2 Morphogenetic Gradients and Positional Information

Biological development uses **morphogen gradients** (concentration gradients of signaling molecules) to provide positional information. We implement analogous mechanisms:

```python
class MorphogenSystem:
    """
    System of morphogenetic signals providing positional information
    
    Enables context-dependent differentiation during development
    """
    def __init__(self, structure: DevelopmentalStructure):
        self.structure = structure
        self.signals = {}
    
    def compute_anterior_posterior_gradient(self) -> np.ndarray:
        """Gradient along primary axis (input → output)"""
        depth = self.structure.depth()
        gradient = np.linspace(0, 1, depth)
        return gradient
    
    def compute_lateral_gradient(self) -> Dict[str, float]:
        """Gradient across parallel pathways at same depth"""
        gradients = {}
        for layer_idx in range(self.structure.depth()):
            modules_at_depth = self.structure.modules_at_depth(layer_idx)
            if len(modules_at_depth) > 1:
                lateral_positions = np.linspace(0, 1, len(modules_at_depth))
                for module, position in zip(modules_at_depth, lateral_positions):
                    gradients[module.id] = position
        return gradients
    
    def compute_information_flow_gradient(self) -> Dict[str, float]:
        """
        Gradient based on graph-theoretic centrality
        
        Modules on critical paths have high values
        """
        centrality = {}
        graph = self.structure.to_graph()
        
        # Compute betweenness centrality
        for module in self.structure.proto_modules:
            centrality[module.id] = betweenness_centrality(graph, module.id)
        
        return centrality
    
    def apply_morphogens(self, module: ProtoModule):
        """
        Use morphogenetic signals to influence module differentiation
        
        High anterior signal → input processing adaptations
        High posterior signal → output generation adaptations
        High centrality → critical path optimization (more capacity)
        """
        ap_position = self.compute_anterior_posterior_gradient()[module.position]
        lateral_position = self.compute_lateral_gradient().get(module.id, 0.5)
        centrality = self.compute_information_flow_gradient()[module.id]
        
        # Anterior modules: larger receptive fields
        if ap_position < 0.3:
            if 'kernel_size' in module.base_hyperparams:
                module.base_hyperparams['kernel_size'] *= 2
        
        # Posterior modules: more abstraction capacity
        if ap_position > 0.7:
            module.base_hyperparams['d_model'] = int(
                module.base_hyperparams['d_model'] * 1.5
            )
        
        # High centrality modules: additional capacity and redundancy
        if centrality > 0.7:
            module.base_hyperparams['d_model'] = int(
                module.base_hyperparams['d_model'] * 1.3
            )
            module.add_redundant_pathways()
```

### 5.3 Recursive Developmental Programs

Hierarchical structures emerge from recursive developmental instructions:

```python
class RecursiveDevelopmentalProgram:
    """
    Developmental programs with self-reference
    
    Enables compact encoding of deep hierarchies:
    - Fractal structures
    - Nested modules
    - Self-similar patterns at multiple scales
    """
    
    def __init__(self, base_module: ModuleGene, recursion_depth: int):
        self.base = base_module
        self.depth = recursion_depth
    
    def unfold(self) -> List[ModuleGene]:
        """
        Recursively expand program into module sequence
        
        Example: Binary tree structure
        depth=0: [A]
        depth=1: [A, A_L, A_R]
        depth=2: [A, A_L, A_L_L, A_L_R, A_R, A_R_L, A_R_R]
        """
        modules = [self.base]
        
        for level in range(self.depth):
            new_modules = []
            for module in modules:
                # Each module spawns children
                left_child = self.create_child(module, "left")
                right_child = self.create_child(module, "right")
                new_modules.extend([left_child, right_child])
            modules.extend(new_modules)
        
        return modules
    
    def create_child(self, parent: ModuleGene, branch: str) -> ModuleGene:
        """Create child module with modified parameters"""
        child = copy.deepcopy(parent)
        child.id = f"{parent.id}_{branch}"
        
        # Scale down capacity at deeper levels
        child.hyperparams['d_model'] = int(parent.hyperparams['d_model'] * 0.8)
        
        return child


class HierarchicalCompositionProgram:
    """
    Hierarchical composition: modules contain sub-modules
    
    Enables evolution of macro-architectures where each module
    is itself a complex learned structure
    """
    
    def __init__(self, macro_genotype: Genotype):
        self.macro = macro_genotype
        self.micro_genotypes = {}  # Maps module IDs to sub-genotypes
    
    def develop_hierarchical(self) -> Phenotype:
        """
        Two-level development:
        1. Develop macro structure
        2. Develop each micro structure
        3. Compose into unified phenotype
        """
        # Develop macro structure
        macro_structure = develop_phenotype_staged(self.macro)
        
        # Develop each micro structure
        micro_phenotypes = {}
        for module_id in macro_structure.modules:
            if module_id in self.micro_genotypes:
                micro = develop_phenotype_staged(self.micro_genotypes[module_id])
                micro_phenotypes[module_id] = micro
        
        # Replace macro modules with developed micro phenotypes
        for module_id, micro_phenotype in micro_phenotypes.items():
            macro_structure.replace_module(module_id, micro_phenotype)
        
        return macro_structure
```

### 5.4 Activity-Dependent Development

Development can be influenced by early experience:

```python
class ActivityDependentDevelopment:
    """
    Developmental plasticity: structure shaped by early activity patterns
    
    Biological analogy: Visual system development requires visual input
    """
    
    def __init__(self, initial_structure: DevelopmentalStructure):
        self.structure = initial_structure
        self.activity_traces = {}
    
    def experience_dependent_pruning(self,
                                    training_data: DataLoader,
                                    pruning_threshold: float = 0.1):
        """
        Prune connections based on early activity
        
        Connections with low activation correlation are pruned
        """
        # Collect activity statistics
        for batch in training_data:
            activations = self.forward_and_record(batch)
            self.update_activity_traces(activations)
        
        # Prune based on activity
        for connection in list(self.structure.connections):
            source_activity = self.activity_traces[connection.source]
            target_activity = self.activity_traces[connection.target]
            
            # Compute correlation
            correlation = np.corrcoef(source_activity, target_activity)[0, 1]
            
            if abs(correlation) < pruning_threshold:
                self.structure.remove_connection(connection)
    
    def activity_dependent_neurogenesis(self,
                                       training_data: DataLoader,
                                       growth_threshold: float = 0.8):
        """
        Add modules to regions of high activity/error
        
        If a module is consistently saturated or error is high,
        add parallel capacity
        """
        error_accumulator = {}
        
        for batch in training_data:
            outputs, errors = self.forward_with_error_tracking(batch)
            for module_id, error in errors.items():
                if module_id not in error_accumulator:
                    error_accumulator[module_id] = []
                error_accumulator[module_id].append(error)
        
        # Add capacity where needed
        for module_id, errors in error_accumulator.items():
            mean_error = np.mean(errors)
            if mean_error > growth_threshold:
                # Duplicate this module (parallel pathway)
                new_module = self.structure.duplicate_module(module_id)
                self.structure.add_module(new_module)
```

### 5.5 Developmental Noise and Canalization

**Stochastic development**: Introducing controlled randomness during morphogenesis:

```python
def develop_with_noise(genotype: Genotype,
                      noise_level: float = 0.1) -> Phenotype:
    """
    Add stochasticity to developmental process
    
    Benefits:
    - Robustness testing (if phenotype succeeds despite noise, it's robust)
    - Ensemble generation (multiple phenotypes from one genotype)
    - Exploration (discover variations on genotype theme)
    """
    structure = specify_basic_structure(genotype)
    
    # Add noise to hyperparameters during proliferation
    for module in structure.proto_modules:
        for param_name, param_value in module.base_hyperparams.items():
            if isinstance(param_value, (int, float)):
                noise = np.random.normal(0, noise_level * abs(param_value))
                module.base_hyperparams[param_name] = param_value + noise
    
    # Stochastic connection realization
    for conn_gene in genotype.connections:
        realization_prob = 1.0 - noise_level
        if random.random() < realization_prob:
            structure.add_connection_from_gene(conn_gene)
    
    # Continue development
    structure = proliferate_modules(structure, genotype.developmental_params)
    structure = differentiate_modules(structure, None)
    phenotype = finalize_phenotype(structure, genotype.plasticity_rules)
    
    return phenotype


class CanalizationMechanism:
    """
    Canalization: buffering against developmental noise
    
    Waddington's concept: developmental pathways are robust to perturbation
    """
    
    def __init__(self, genotype: Genotype, num_samples: int = 10):
        self.genotype = genotype
        self.num_samples = num_samples
    
    def measure_canalization(self) -> float:
        """
        Measure how consistent phenotypes are across stochastic development
        
        Low variance → high canalization (robust development)
        High variance → low canalization (sensitive to noise)
        """
        phenotypes = []
        for _ in range(self.num_samples):
            pheno = develop_with_noise(self.genotype, noise_level=0.2)
            phenotypes.append(pheno)
        
        # Measure phenotypic variance
        behavioral_vectors = [extract_behavior(p) for p in phenotypes]
        variance = np.var(behavioral_vectors, axis=0).mean()
        
        canalization_score = 1.0 / (1.0 + variance)  # Higher score = more canalized
        return canalization_score
```

---

## 6. Multi-Scale Learning Mechanisms

The framework integrates three nested timescales of adaptation:

1. **Evolutionary timescale**: Genotype optimization (slowest)
2. **Developmental timescale**: Phenotype construction (intermediate)
3. **Experiential timescale**: Parameter learning (fastest)

### 6.1 Intra-Lifetime Learning

Within a single lifetime (task episode), phenotypes learn through:

#### 6.1.1 Gradient-Based Learning

Standard backpropagation through time/space:

```python
class GradientLearner:
    """Standard SGD-based learning within phenotype lifetime"""
    
    def __init__(self, phenotype: Phenotype, learning_rate: float = 1e-3):
        self.phenotype = phenotype
        self.optimizer = torch.optim.Adam(
            phenotype.parameters(),
            lr=learning_rate
        )
    
    def learn_from_experience(self,
                             data: DataLoader,
                             num_epochs: int = 10) -> float:
        """
        Train phenotype on task data
        
        Returns final loss (fitness proxy)
        """
        for epoch in range(num_epochs):
            epoch_loss = 0.0
            for batch in data:
                inputs, targets = batch
                
                # Forward pass
                outputs = self.phenotype.forward(inputs)
                loss = compute_loss(outputs, targets)
                
                # Backward pass
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item()
            
            epoch_loss /= len(data)
            
        return epoch_loss
```

#### 6.1.2 Local Plasticity Rules

Hebbian and other local learning rules (no backpropagation):

```python
class LocalPlasticityLearner:
    """
    Learning via local synaptic plasticity rules
    
    Does not require backpropagation or global error signal
    """
    
    def __init__(self, phenotype: Phenotype):
        self.phenotype = phenotype
    
    def apply_plasticity_rules(self,
                              pre_activations: Dict[str, torch.Tensor],
                              post_activations: Dict[str, torch.Tensor],
                              learning_signals: Dict[str, float]):
        """
        Update connection strengths based on local activity
        
        Args:
            pre_activations: Activity of presynaptic modules
            post_activations: Activity of postsynaptic modules
            learning_signals: Neuromodulatory signals (reward, novelty, etc.)
        """
        for conn_id, plasticity_rule in self.phenotype.plasticity.items():
            # Get relevant activations
            pre = pre_activations[plasticity_rule.source_id]
            post = post_activations[plasticity_rule.target_id]
            
            # Apply learning rule
            if plasticity_rule.rule_type == PlasticityType.HEBBIAN:
                delta_w = self.hebbian_update(pre, post, plasticity_rule.metaparams)
            
            elif plasticity_rule.rule_type == PlasticityType.OJA:
                delta_w = self.oja_update(pre, post, plasticity_rule.metaparams)
            
            elif plasticity_rule.rule_type == PlasticityType.STDP:
                delta_w = self.stdp_update(pre, post, plasticity_rule.metaparams)
            
            elif plasticity_rule.rule_type == PlasticityType.DOPAMINE_MODULATED:
                reward_signal = learning_signals.get('reward', 0.0)
                delta_w = self.reward_modulated_update(
                    pre, post, reward_signal, plasticity_rule.metaparams
                )
            
            # Apply neuromodulation if present
            if plasticity_rule.modulation:
                modulation_factor = plasticity_rule.modulation.compute(
                    self.phenotype.get_state(),
                    learning_signals
                )
                delta_w *= modulation_factor
            
            # Update weights
            connection = self.phenotype.graph.get_connection(conn_id)
            connection.weights += delta_w
    
    def hebbian_update(self,
                      pre: torch.Tensor,
                      post: torch.Tensor,
                      metaparams: Dict) -> torch.Tensor:
        """
        Hebbian learning: Δw = η · pre · post
        
        "Neurons that fire together, wire together"
        """
        eta = metaparams['η']
        delta_w = eta * torch.outer(post, pre)
        return delta_w
    
    def oja_update(self,
                  pre: torch.Tensor,
                  post: torch.Tensor,
                  metaparams: Dict) -> torch.Tensor:
        """
        Oja's rule: Hebbian with weight normalization
        
        Δw = η · (pre · post - post² · w)
        
        Prevents unlimited weight growth
        """
        eta = metaparams['η']
        w_current = self.get_current_weights()
        
        delta_w = eta * (torch.outer(post, pre) - 
                        (post ** 2).unsqueeze(1) * w_current)
        return delta_w
    
    def stdp_update(self,
                   pre_spikes: torch.Tensor,
                   post_spikes: torch.Tensor,
                   metaparams: Dict) -> torch.Tensor:
        """
        Spike-timing dependent plasticity
        
        Δw depends on temporal order of pre/post spikes
        """
        tau_plus = metaparams['τ']
        tau_minus = metaparams['τ']
        A_plus = metaparams['A_plus']
        A_minus = metaparams['A_minus']
        
        # Compute spike time differences
        time_diffs = post_spikes.unsqueeze(1) - pre_spikes.unsqueeze(0)
        
        # LTP: pre before post (positive time diff)
        ltp = A_plus * torch.exp(-time_diffs / tau_plus) * (time_diffs > 0)
        
        # LTD: post before pre (negative time diff)
        ltd = -A_minus * torch.exp(time_diffs / tau_minus) * (time_diffs < 0)
        
        delta_w = ltp + ltd
        return delta_w
    
    def reward_modulated_update(self,
                               pre: torch.Tensor,
                               post: torch.Tensor,
                               reward: float,
                               metaparams: Dict) -> torch.Tensor:
        """
        Dopamine-modulated plasticity
        
        Hebbian update scaled by reward signal (temporal difference error)
        """
        eta = metaparams['η']
        eligibility_trace = torch.outer(post, pre)
        
        # Reward prediction error modulates plasticity
        delta_w = eta * reward * eligibility_trace
        
        return delta_w
```

#### 6.1.3 Meta-Learned Plasticity

Learning rules themselves can be learned:

```python
class MetaLearnedPlasticity:
    """
    Plasticity rules with learnable parameters
    
    Based on "Backpropamine" (Miconi et al., 2018)
    """
    
    def __init__(self, phenotype: Phenotype):
        self.phenotype = phenotype
        
        # Learnable plasticity coefficients (evolved or meta-learned)
        self.plasticity_params = nn.ParameterDict({
            conn_id: nn.Parameter(torch.randn(conn.weight.shape))
            for conn_id, conn in phenotype.graph.connections.items()
        })
    
    def forward_with_plasticity(self,
                               x: torch.Tensor,
                               learning_signal: float = 1.0) -> torch.Tensor:
        """
        Forward pass with online plasticity
        
        Weights are modified during forward pass based on activations
        """
        activations = {'input': x}
        
        for layer_id in self.phenotype.graph.execution_order:
            # Get inputs
            inputs = [activations[pred] for pred in self.phenotype.graph.predecessors(layer_id)]
            aggregated = torch.cat(inputs, dim=-1)
            
            # Get base connection weights
            connection = self.phenotype.graph.get_connection_to(layer_id)
            base_weights = connection.weights
            
            # Compute plastic component
            pre_activity = aggregated
            post_activity = activations.get(layer_id, torch.zeros_like(aggregated))
            
            # Learnable plasticity rule: Δw = α · Hebbian + β · anti-Hebbian
            alpha = self.plasticity_params[layer_id][..., 0]
            beta = self.plasticity_params[layer_id][..., 1]
            
            plastic_weights = (alpha * torch.outer(post_activity, pre_activity) +
                             beta * torch.outer(post_activity, pre_activity) * -1)
            
            # Effective weights = base + plastic · learning_signal
            effective_weights = base_weights + learning_signal * plastic_weights
            
            # Apply layer computation
            activations[layer_id] = self.phenotype.modules[layer_id](
                torch.matmul(effective_weights, aggregated)
            )
        
        return activations['output']
```

### 6.2 Ontogenetic Learning (Developmental)

Learning that occurs during development itself:

```python
class DevelopmentalLearning:
    """
    Learning during morphogenesis
    
    Structure is shaped by early experience before "birth" (deployment)
    """
    
    def __init__(self, genotype: Genotype, pretraining_data: DataLoader):
        self.genotype = genotype
        self.pretraining_data = pretraining_data
    
    def develop_with_pretraining(self) -> Phenotype:
        """
        Interleave development stages with learning
        
        Biological analogy: prenatal learning (fetuses respond to sound)
        """
        # Initial specification
        structure = specify_basic_structure(self.genotype)
        
        # Proliferation
        structure = proliferate_modules(structure, self.genotype.developmental_params)
        
        # Partial instantiation for early learning
        partial_phenotype = instantiate_partial(structure)
        
        # Pretrain on developmental data
        self.pretrain_phase(partial_phenotype, self.pretraining_data, epochs=5)
        
        # Activity-dependent pruning based on pretrained activations
        structure = prune_based_on_activity(structure, partial_phenotype)
        
        # Continue development
        structure = differentiate_modules(structure, None)
        
        # Another learning phase
        partial_phenotype = update_from_structure(partial_phenotype, structure)
        self.pretrain_phase(partial_phenotype, self.pretraining_data, epochs=5)
        
        # Final maturation
        phenotype = finalize_phenotype(structure, self.genotype.plasticity_rules)
        
        # Transfer learned weights
        phenotype.load_weights_from(partial_phenotype)
        
        return phenotype
    
    def pretrain_phase(self, phenotype: Phenotype, data: DataLoader, epochs: int):
        """Quick pretraining during development"""
        optimizer = torch.optim.SGD(phenotype.parameters(), lr=0.01)
        
        for epoch in range(epochs):
            for batch in data:
                inputs, targets = batch
                outputs = phenotype.forward(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
```

### 6.3 Phylogenetic Learning (Evolutionary)

Evolution as learning across generations:

```python
class EvolutionaryLearning:
    """
    Conceptualize evolution as population-level learning
    
    The population "learns" about the task distribution through
    selection pressure acting as a learning signal
    """
    
    def __init__(self, task_distribution: TaskDistribution):
        self.task_distribution = task_distribution
        self.population_memory = []  # Archive of successful genotypes
    
    def compute_evolutionary_gradient(self,
                                     population: List[Genotype],
                                     fitnesses: np.ndarray) -> Dict:
        """
        Estimate gradient of expected fitness w.r.t. genotype distribution
        
        This enables gradient-guided evolution (hybridizing evolution and gradient descent)
        """
        # Fitness-weighted average genotype (mean of distribution)
        fitness_weights = softmax(fitnesses)
        
        mean_genotype = self.weighted_average_genotype(population, fitness_weights)
        
        # Variance (spread of distribution)
        variance = self.compute_genotype_variance(population, mean_genotype)
        
        # Gradient estimate: direction of fitness improvement
        gradient = {
            'mean_direction': self.estimate_mean_gradient(population, fitnesses),
            'variance_direction': self.estimate_variance_gradient(population, fitnesses)
        }
        
        return gradient
    
    def evolutionary_gradient_ascent(self,
                                    population: List[Genotype],
                                    gradient: Dict,
                                    step_size: float = 0.1) -> List[Genotype]:
        """
        Update population using estimated gradient
        
        Combines evolutionary search with gradient information
        """
        updated_population = []
        
        for genotype in population:
            # Move genotype in direction of gradient
            updated = self.apply_gradient_to_genotype(
                genotype,
                gradient['mean_direction'],
                step_size
            )
            updated_population.append(updated)
        
        return updated_population
    
    def apply_gradient_to_genotype(self,
                                   genotype: Genotype,
                                   gradient: Dict,
                                   step_size: float) -> Genotype:
        """
        Apply gradient update to genotype hyperparameters
        
        Only applicable to continuous parameters
        """
        updated = copy.deepcopy(genotype)
        
        for module in updated.modules:
            for param_name in module.hyperparams:
                if param_name in gradient and isinstance(module.hyperparams[param_name], float):
                    module.hyperparams[param_name] += step_size * gradient[param_name]
        
        return updated
```

### 6.4 Multi-Timescale Integration

The three timescales interact:

```python
class MultiTimescaleAdaptation:
    """
    Integrate evolutionary, developmental, and experiential learning
    
    Key insight: Different timescales handle different types of adaptation
    - Evolution: discovers learning rules and architectural priors
    - Development: builds task-appropriate structure
    - Experience: fine-tunes parameters for specific instances
    """
    
    def __init__(self,
                 population_size: int = 100,
                 num_generations: int = 500):
        self.population_size = population_size
        self.num_generations = num_generations
    
    def adapt(self, task_distribution: TaskDistribution) -> Genotype:
        """
        Complete multi-timescale adaptation cycle
        
        Returns best-adapted genotype after evolutionary search
        """
        # Initialize population
        population = initialize_population(self.population_size)
        
        for generation in range(self.num_generations):
            # For each genotype in population
            fitnesses = []
            
            for genotype in population:
                # DEVELOPMENTAL TIMESCALE: Construct phenotype
                phenotype = develop_phenotype_staged(genotype)
                
                # EXPERIENTIAL TIMESCALE: Learn on task samples
                generation_fitness = 0.0
                for task in task_distribution.sample(num_tasks=10):
                    # Learn from task data
                    task_data = task.get_training_data()
                    learner = GradientLearner(phenotype, learning_rate=1e-3)
                    final_loss = learner.learn_from_experience(task_data, num_epochs=5)
                    
                    # Evaluate on test data
                    test_data = task.get_test_data()
                    test_performance = evaluate_phenotype(phenotype, test_data)
                    
                    generation_fitness += test_performance
                
                fitnesses.append(generation_fitness / 10)
            
            fitnesses = np.array(fitnesses)
            
            # EVOLUTIONARY TIMESCALE: Select and reproduce
            parents = select_parents(population, fitnesses, selection_size=self.population_size)
            
            offspring = []
            for _ in range(self.population_size):
                if random.random() < 0.7:  # Crossover
                    p1, p2 = random.sample(parents, 2)
                    child, _ = modular_crossover(p1, p2)
                else:  # Mutation
                    parent = random.choice(parents)
                    child = copy.deepcopy(parent)
                
                # Apply mutations
                child = mutate_hyperparameters(child)
                child = mutate_plasticity(child)
                
                offspring.append(child)
            
            population = offspring
        
        # Return best genotype from final population
        final_fitnesses = [self.evaluate_genotype(g, task_distribution) 
                          for g in population]
        best_idx = np.argmax(final_fitnesses)
        return population[best_idx]
    
    def evaluate_genotype(self, genotype: Genotype, 
                         task_distribution: TaskDistribution) -> float:
        """Full evaluation of genotype across task distribution"""
        phenotype = develop_phenotype_staged(genotype)
        
        total_fitness = 0.0
        num_eval_tasks = 20
        
        for task in task_distribution.sample(num_tasks=num_eval_tasks):
            learner = GradientLearner(phenotype, learning_rate=1e-3)
            train_data = task.get_training_data()
            learner.learn_from_experience(train_data, num_epochs=10)
            
            test_data = task.get_test_data()
            performance = evaluate_phenotype(phenotype, test_data)
            total_fitness += performance
        
        return total_fitness / num_eval_tasks
```

### 6.5 Baldwin Effect and Genetic Assimilation

**Baldwin Effect**: Learned behaviors can guide evolution

```python
class BaldwinianEvolution:
    """
    Learning influences evolutionary trajectory
    
    Genotypes that learn faster have higher fitness, creating
    selection pressure toward architectures with good learning dynamics
    """
    
    def __init__(self):
        self.learning_speed_weight = 0.5  # How much to weight learning speed vs final performance
    
    def baldwinian_fitness(self,
                          genotype: Genotype,
                          task: Task) -> float:
        """
        Fitness combines:
        1. Final performance after learning
        2. Speed of learning (sample efficiency)
        """
        phenotype = develop_phenotype_staged(genotype)
        
        # Track performance during learning
        performance_curve = []
        learner = GradientLearner(phenotype)
        
        for epoch in range(50):
            train_loss = learner.learn_from_experience(
                task.get_training_data(),
                num_epochs=1
            )
            
            test_perf = evaluate_phenotype(phenotype, task.get_test_data())
            performance_curve.append(test_perf)
        
        # Final performance
        final_performance = performance_curve[-1]
        
        # Learning speed (area under learning curve)
        learning_speed = np.trapz(performance_curve)
        
        # Combined fitness
        fitness = (self.learning_speed_weight * learning_speed +
                  (1 - self.learning_speed_weight) * final_performance)
        
        return fitness


class GeneticAssimilation:
    """
    Learned behaviors become innate through evolution
    
    If a behavior is consistently learned, evolution can hardcode it
    into the architecture, freeing up learning capacity
    """
    
    def __init__(self, population: List[Genotype]):
        self.population = population
        self.behavioral_statistics = {}
    
    def detect_universal_behaviors(self, task_distribution: TaskDistribution):
        """
        Identify behaviors that are learned by all individuals
        
        These are candidates for genetic assimilation
        """
        # Collect learned behaviors across population
        for genotype in self.population:
            phenotype = develop_phenotype_staged(genotype)
            
            # Learn on tasks
            for task in task_distribution.sample(num_tasks=10):
                learner = GradientLearner(phenotype)
                learner.learn_from_experience(task.get_training_data())
                
                # Extract learned computations
                behaviors = extract_learned_behaviors(phenotype)
                
                for behavior_id, behavior_params in behaviors.items():
                    if behavior_id not in self.behavioral_statistics:
                        self.behavioral_statistics[behavior_id] = []
                    self.behavioral_statistics[behavior_id].append(behavior_params)
        
        # Identify universal behaviors (learned by >90% of population)
        universal_behaviors = {}
        for behavior_id, param_list in self.behavioral_statistics.items():
            if len(param_list) > 0.9 * len(self.population):
                # Compute mean learned parameters
                mean_params = np.mean(param_list, axis=0)
                universal_behaviors[behavior_id] = mean_params
        
        return universal_behaviors
    
    def assimilate_behaviors(self,
                           genotype: Genotype,
                           universal_behaviors: Dict) -> Genotype:
        """
        Hardcode universal behaviors into genotype
        
        Moves them from learned (plastic) to innate (hardwired)
        """
        assimilated = copy.deepcopy(genotype)
        
        for behavior_id, params in universal_behaviors.items():
            # Add new module or connection encoding this behavior
            if behavior_id.startswith('attention_pattern'):
                # Hardcode learned attention pattern
                new_module = ModuleGene(
                    id=f"innate_{behavior_id}",
                    type=ModuleType.SPARSE_ATTENTION,
                    hyperparams={'pattern': params, 'd_model': 512},
                    activation='softmax',
                    normalization='layer_norm',
                    initialization='fixed'  # Fixed to learned parameters
                )
                assimilated.modules.append(new_module)
            
            elif behavior_id.startswith('feature_transform'):
                # Hardcode learned transformation
                new_connection = ConnectionGene(
                    source='input',
                    target=f"innate_{behavior_id}",
                    weight=1.0,
                    properties=ConnectionProperties(
                        fixed_transform=params  # Hardwired transformation
                    )
                )
                assimilated.connections.append(new_connection)
        
        return assimilated
```

---

## 7. Population-Based Meta-Optimization

### 7.1 Population Dynamics

Evolution operates on populations, not individuals:

```python
class PopulationDynamics:
    """
    Model population-level phenomena:
    - Genetic diversity maintenance
    - Speciation
    - Co-evolution
    - Ecosystem dynamics
    """
    
    def __init__(self, population_size: int = 200):
        self.population_size = population_size
        self.species = []  # List of species (clusters of similar genotypes)
    
    def maintain_diversity(self,
                          population: List[Genotype],
                          diversity_threshold: float = 0.3):
        """
        Actively prevent premature convergence
        
        Methods:
        - Fitness sharing (penalize crowding)
        - Speciation (protect niches)
        - Novelty search (reward exploration)
        """
        # Compute diversity metrics
        diversity = self.compute_population_diversity(population)
        
        if diversity < diversity_threshold:
            # Apply diversity-preserving operators
            
            # 1. Increase mutation rate temporarily
            self.adaptive_mutation_rate = min(0.8, self.adaptive_mutation_rate * 1.5)
            
            # 2. Inject random genotypes
            num_random = int(0.1 * self.population_size)
            for _ in range(num_random):
                random_genotype = initialize_random_genotype()
                population[np.random.randint(len(population))] = random_genotype
            
            # 3. Apply speciation to protect niches
            self.speciate(population)
        
        else:
            # Reduce mutation rate if diversity is healthy
            self.adaptive_mutation_rate = max(0.1, self.adaptive_mutation_rate * 0.95)
    
    def compute_population_diversity(self, population: List[Genotype]) -> float:
        """
        Measure genetic diversity
        
        Average pairwise distance between individuals
        """
        if len(population) < 2:
            return 0.0
        
        total_distance = 0.0
        num_pairs = 0
        
        for i in range(len(population)):
            for j in range(i + 1, len(population)):
                distance = genotype_distance(population[i], population[j])
                total_distance += distance
                num_pairs += 1
        
        diversity = total_distance / num_pairs
        return diversity
    
    def speciate(self, population: List[Genotype], compatibility_threshold: float = 5.0):
        """
        Divide population into species (NEAT-style speciation)
        
        Species = clusters of genetically similar individuals
        """
        self.species = []
        
        for genotype in population:
            # Try to assign to existing species
            assigned = False
            for species in self.species:
                representative = species.representative
                distance = genotype_distance(genotype, representative)
                
                if distance < compatibility_threshold:
                    species.add_member(genotype)
                    assigned = True
                    break
            
            # Create new species if no match
            if not assigned:
                new_species = Species(representative=genotype)
                new_species.add_member(genotype)
                self.species.append(new_species)
        
        # Adjust fitness based on species size (fitness sharing)
        for species in self.species:
            species.adjust_fitness_sharing()
    
    def allocate_offspring(self, species_fitnesses: List[float]) -> List[int]:
        """
        Allocate offspring slots to species proportional to fitness
        
        Maintains diverse species even if some are currently less fit
        """
        total_fitness = sum(species_fitnesses)
        
        offspring_allocation = []
        for species_fitness in species_fitnesses:
            # Proportional allocation
            proportion = species_fitness / total_fitness if total_fitness > 0 else 1.0 / len(species_fitnesses)
            num_offspring = int(proportion * self.population_size)
            
            # Ensure minimum allocation (protect weak species temporarily)
            num_offspring = max(2, num_offspring)
            
            offspring_allocation.append(num_offspring)
        
        # Adjust to exactly population_size
        while sum(offspring_allocation) > self.population_size:
            max_idx = np.argmax(offspring_allocation)
            offspring_allocation[max_idx] -= 1
        
        while sum(offspring_allocation) < self.population_size:
            max_fitness_idx = np.argmax(species_fitnesses)
            offspring_allocation[max_fitness_idx] += 1
        
        return offspring_allocation


class Species:
    """A species: cluster of similar genotypes"""
    
    def __init__(self, representative: Genotype):
        self.representative = representative
        self.members = []
        self.fitness_history = []
        self.age = 0
        self.stagnation_counter = 0
    
    def add_member(self, genotype: Genotype):
        self.members.append(genotype)
    
    def adjust_fitness_sharing(self):
        """
        Apply fitness sharing within species
        
        Reduces fitness of individuals in crowded species
        """
        species_size = len(self.members)
        for member in self.members:
            member.shared_fitness = member.fitness / species_size
    
    def update_age(self, current_best_fitness: float):
        """
        Track species age and stagnation
        
        Species that don't improve get removed
        """
        self.age += 1
        
        if current_best_fitness > max(self.fitness_history[-5:], default=0):
            self.stagnation_counter = 0
        else:
            self.stagnation_counter += 1
        
        self.fitness_history.append(current_best_fitness)
    
    def is_stagnant(self, threshold: int = 15) -> bool:
        """Species hasn't improved in threshold generations"""
        return self.stagnation_counter > threshold
```

### 7.2 Co-Evolution and Competitive Dynamics

Multiple populations evolve together:

```python
class CoEvolutionarySystem:
    """
    Co-evolution of multiple populations
    
    Applications:
    - Architecture population + Task population (evolving both)
    - Competitive co-evolution (adversarial training)
    - Symbiotic co-evolution (modular composition)
    """
    
    def __init__(self):
        self.populations = {}
    
    def competitive_coevolution(self,
                               population_A: List[Genotype],
                               population_B: List[Genotype],
                               num_generations: int = 500):
        """
        Arms race: populations evolve to beat each other
        
        Example: Architectures vs. Adversarial perturbations
        """
        for generation in range(num_generations):
            # Evaluate A against B
            fitnesses_A = []
            for genotype_A in population_A:
                fitness = 0.0
                for genotype_B in random.sample(population_B, k=10):
                    # Competitive evaluation
                    score = self.compete(genotype_A, genotype_B)
                    fitness += score
                fitnesses_A.append(fitness / 10)
            
            # Evaluate B against A
            fitnesses_B = []
            for genotype_B in population_B:
                fitness = 0.0
                for genotype_A in random.sample(population_A, k=10):
                    score = self.compete(genotype_B, genotype_A)
                    fitness += score
                fitnesses_B.append(fitness / 10)
            
            # Evolve both populations
            population_A = self.evolve_generation(population_A, fitnesses_A)
            population_B = self.evolve_generation(population_B, fitnesses_B)
        
        return population_A, population_B
    
    def cooperative_coevolution(self,
                               subpopulations: Dict[str, List[Genotype]],
                               num_generations: int = 500):
        """
        Cooperative co-evolution: subpopulations evolve complementary skills
        
        Example: Modular architectures where each subpopulation evolves
        a different functional module
        """
        for generation in range(num_generations):
            # For each subpopulation
            for subpop_name, subpopulation in subpopulations.items():
                fitnesses = []
                
                for genotype in subpopulation:
                    # Evaluate in combination with representatives from other subpopulations
                    collaborators = {
                        name: random.choice(pop) 
                        for name, pop in subpopulations.items() 
                        if name != subpop_name
                    }
                    collaborators[subpop_name] = genotype
                    
                    # Compose complete system from modules
                    composite_system = self.compose_modules(collaborators)
                    
                    # Evaluate composite system
                    fitness = self.evaluate_composite(composite_system)
                    fitnesses.append(fitness)
                
                # Evolve this subpopulation
                subpopulations[subpop_name] = self.evolve_generation(
                    subpopulation,
                    fitnesses
                )
        
        return subpopulations
    
    def compose_modules(self, module_genotypes: Dict[str, Genotype]) -> Genotype:
        """
        Compose complete architecture from evolved modules
        
        Each module handles a different aspect (vision, memory, reasoning, etc.)
        """
        composite = Genotype(modules=[], connections=[], plasticity_rules=[], developmental_params=None)
        
        # Add all modules
        for module_name, genotype in module_genotypes.items():
            for module in genotype.modules:
                module.id = f"{module_name}_{module.id}"
                composite.modules.append(module)
        
        # Connect modules according to composition protocol
        composite.connections = self.define_inter_module_connections(module_genotypes)
        
        # Merge plasticity rules
        for genotype in module_genotypes.values():
            composite.plasticity_rules.extend(genotype.plasticity_rules)
        
        # Use developmental params from primary module
        composite.developmental_params = list(module_genotypes.values())[0].developmental_params
        
        return composite
```

### 7.3 Island Model and Migration

Geographic structure in evolution:

```python
class IslandModel:
    """
    Island model: multiple isolated populations with occasional migration
    
    Benefits:
    - Explores diverse regions of genotype space
    - Prevents premature convergence
    - Discovers multiple solutions
    """
    
    def __init__(self, num_islands: int = 10, island_size: int = 50):
        self.num_islands = num_islands
        self.island_size = island_size
        self.islands = [[] for _ in range(num_islands)]
    
    def evolve_islands(self,
                      task_distribution: TaskDistribution,
                      num_generations: int = 1000,
                      migration_interval: int = 50,
                      migration_rate: float = 0.05):
        """
        Evolve multiple islands in parallel with periodic migration
        """
        # Initialize islands
        for i in range(self.num_islands):
            self.islands[i] = initialize_population(self.island_size)
        
        for generation in range(num_generations):
            # Evolve each island independently
            for island_idx in range(self.num_islands):
                island = self.islands[island_idx]
                
                # Evaluate fitness
                fitnesses = evaluate_population(island, task_distribution)
                
                # Select and reproduce
                parents = select_parents(island, fitnesses, selection_size=self.island_size)
                offspring = generate_offspring(parents, offspring_size=self.island_size)
                
                # Update island
                self.islands[island_idx] = offspring
            
            # Periodic migration
            if generation % migration_interval == 0:
                self.migrate(migration_rate)
        
        # Collect best from all islands
        all_genotypes = [g for island in self.islands for g in island]
        return all_genotypes
    
    def migrate(self, migration_rate: float):
        """
        Exchange individuals between neighboring islands
        
        Topology: Ring (each island connected to two neighbors)
        """
        num_migrants = int(self.island_size * migration_rate)
        
        for i in range(self.num_islands):
            # Select migrants from this island
            migrants = random.sample(self.islands[i], num_migrants)
            
            # Send to neighboring island
            neighbor_idx = (i + 1) % self.num_islands
            
            # Replace worst individuals in neighbor with migrants
            neighbor_fitnesses = [self.quick_fitness_estimate(g) for g in self.islands[neighbor_idx]]
            worst_indices = np.argsort(neighbor_fitnesses)[:num_migrants]
            
            for migrant, worst_idx in zip(migrants, worst_indices):
                self.islands[neighbor_idx][worst_idx] = copy.deepcopy(migrant)
    
    def quick_fitness_estimate(self, genotype: Genotype) -> float:
        """Fast fitness proxy for migration decisions"""
        # Use cached fitness or quick evaluation
        if hasattr(genotype, 'cached_fitness'):
            return genotype.cached_fitness
        else:
            # Quick evaluation on small sample
            return self.evaluate_on_sample(genotype)
```

### 7.4 Quality Diversity Algorithms

Beyond fitness: discover diverse high-quality solutions:

```python
class MAPElites:
    """
    MAP-Elites: Illumination algorithm
    
    Discovers diverse high-performing solutions across behavioral space
    Creates a map: behavior → best genotype with that behavior
    """
    
    def __init__(self, behavior_dimensions: List[Tuple[str, float, float]]):
        """
        Args:
            behavior_dimensions: List of (dimension_name, min_value, max_value)
            Example: [('depth', 1, 50), ('width', 64, 2048), ('parameter_count', 1e6, 1e9)]
        """
        self.behavior_dimensions = behavior_dimensions
        self.map = {}  # Maps behavior descriptor to (genotype, fitness)
        self.grid_resolution = 20  # Discretization of each dimension
    
    def run(self,
           task_distribution: TaskDistribution,
           num_iterations: int = 10000):
        """
        Iteratively fill the behavior-fitness map
        """
        # Initialize with random genotypes
        for _ in range(100):
            genotype = initialize_random_genotype()
            self.add_to_map(genotype, task_distribution)
        
        for iteration in range(num_iterations):
            # Sample random genotype from map
            genotype = self.sample_from_map()
            
            # Mutate
            offspring = mutate_hyperparameters(genotype)
            offspring = add_connection_mutation(offspring)
            offspring = mutate_plasticity(offspring)
            
            # Evaluate and add to map
            self.add_to_map(offspring, task_distribution)
            
            if iteration % 100 == 0:
                coverage = len(self.map) / (self.grid_resolution ** len(self.behavior_dimensions))
                print(f"Iteration {iteration}: Map coverage = {coverage:.2%}")
        
        return self.map
    
    def add_to_map(self, genotype: Genotype, task_distribution: TaskDistribution):
        """
        Add genotype to map if it's the best in its behavioral niche
        """
        # Extract behavioral characterization
        behavior = self.extract_behavior_descriptor(genotype)
        
        # Discretize to grid cell
        cell = self.discretize_behavior(behavior)
        
        # Evaluate fitness
        fitness = self.evaluate_genotype(genotype, task_distribution)
        
        # Add if cell empty or better than existing
        if cell not in self.map or fitness > self.map[cell][1]:
            self.map[cell] = (genotype, fitness)
    
    def extract_behavior_descriptor(self, genotype: Genotype) -> np.ndarray:
        """
        Extract behavioral features
        
        Examples:
        - Architecture properties: depth, width, connectivity
        - Computational properties: FLOPs, memory, latency
        - Functional properties: attention patterns, recurrence
        """
        phenotype = develop_phenotype_staged(genotype)
        
        behavior = []
        for dim_name, _, _ in self.behavior_dimensions:
            if dim_name == 'depth':
                behavior.append(len(phenotype.graph.execution_order))
            elif dim_name == 'width':
                behavior.append(np.mean([m.hyperparams.get('d_model', 0) 
                                        for m in genotype.modules]))
            elif dim_name == 'parameter_count':
                behavior.append(sum(p.numel() for p in phenotype.parameters()))
            elif dim_name == 'attention_ratio':
                attention_modules = sum(1 for m in genotype.modules 
                                       if 'attention' in m.type.value.lower())
                behavior.append(attention_modules / len(genotype.modules))
            elif dim_name == 'recurrence':
                has_recurrence = any(c.properties.recurrent for c in genotype.connections)
                behavior.append(1.0 if has_recurrence else 0.0)
        
        return np.array(behavior)
    
    def discretize_behavior(self, behavior: np.ndarray) -> Tuple:
        """Map continuous behavior to discrete grid cell"""
        cell = []
        for value, (dim_name, min_val, max_val) in zip(behavior, self.behavior_dimensions):
            # Normalize to [0, 1]
            normalized = (value - min_val) / (max_val - min_val)
            normalized = np.clip(normalized, 0, 1)
            
            # Discretize
            bin_idx = int(normalized * (self.grid_resolution - 1))
            cell.append(bin_idx)
        
        return tuple(cell)
    
    def sample_from_map(self) -> Genotype:
        """Sample random genotype from current map"""
        cell = random.choice(list(self.map.keys()))
        genotype, _ = self.map[cell]
        return copy.deepcopy(genotype)
    
    def get_diversity_metrics(self) -> Dict:
        """Compute quality-diversity metrics"""
        if not self.map:
            return {'coverage': 0, 'qd_score': 0, 'max_fitness': 0}
        
        # Coverage: fraction of cells filled
        total_cells = self.grid_resolution ** len(self.behavior_dimensions)
        coverage = len(self.map) / total_cells
        
        # QD-Score: sum of all fitness values
        qd_score = sum(fitness for _, fitness in self.map.values())
        
        # Max fitness
        max_fitness = max(fitness for _, fitness in self.map.values())
        
        return {
            'coverage': coverage,
            'qd_score': qd_score,
            'max_fitness': max_fitness
        }
```

---

## 8. Computational Implementation

### 8.1 Efficiency Considerations

Evolving neural architectures is computationally expensive. We employ several optimizations:

#### 8.1.1 Surrogate-Based Fitness Estimation

```python
class SurrogateFitnessModel:
    """
    Learn to predict fitness from genotype features without full evaluation
    
    Reduces evaluation cost by orders of magnitude
    """
    
    def __init__(self):
        self.model = GradientBoostingRegressor(n_estimators=100)
        self.training_data = []
    
    def add_evaluation(self, genotype: Genotype, true_fitness: float):
        """Add observed genotype-fitness pair to training set"""
        features = self.extract_genotype_features(genotype)
        self.training_data.append((features, true_fitness))
        
        # Retrain periodically
        if len(self.training_data) % 50 == 0:
            self.retrain()
    
    def predict_fitness(self, genotype: Genotype) -> Tuple[float, float]:
        """
        Predict fitness without evaluation
        
        Returns: (predicted_fitness, uncertainty)
        """
        features = self.extract_genotype_features(genotype)
        
        if len(self.training_data) < 10:
            # Not enough data, return high uncertainty
            return 0.0, float('inf')
        
        # Prediction
        predicted = self.model.predict([features])[0]
        
        # Uncertainty estimation (using ensemble variance)
        predictions = [estimator.predict([features])[0] 
                      for estimator in self.model.estimators_]
        uncertainty = np.std(predictions)
        
        return predicted, uncertainty
    
    def extract_genotype_features(self, genotype: Genotype) -> np.ndarray:
        """
        Extract features predictive of fitness
        
        Features include:
        - Structural: depth, width, connectivity density
        - Compositional: module type distribution
        - Developmental: growth parameters
        - Plasticity: learning rule types
        """
        features = []
        
        # Structural features
        features.append(len(genotype.modules))
        features.append(len(genotype.connections))
        features.append(len(genotype.connections) / max(1, len(genotype.modules)))  # density
        
        # Module type distribution
        module_type_counts = Counter([m.type for m in genotype.modules])
        for module_type in ModuleType:
            features.append(module_type_counts.get(module_type, 0))
        
        # Average hyperparameters
        avg_d_model = np.mean([m.hyperparams.get('d_model', 0) for m in genotype.modules])
        features.append(avg_d_model)
        
        # Developmental parameters
        features.append(genotype.developmental_params.base_layers)
        features.append(genotype.developmental_params.width_multiplier)
        features.append(genotype.developmental_params.initial_connection_density)
        
        # Plasticity
        features.append(len(genotype.plasticity_rules))
        plasticity_type_counts = Counter([p.rule_type for p in genotype.plasticity_rules])
        for plasticity_type in PlasticityType:
            features.append(plasticity_type_counts.get(plasticity_type, 0))
        
        return np.array(features)
    
    def retrain(self):
        """Retrain surrogate model on accumulated data"""
        if len(self.training_data) < 10:
            return
        
        X = np.array([features for features, _ in self.training_data])
        y = np.array([fitness for _, fitness in self.training_data])
        
        self.model.fit(X, y)


class AcquisitionFunction:
    """
    Decide which genotypes to evaluate next
    
    Balance exploration (high uncertainty) vs exploitation (high predicted fitness)
    """
    
    def __init__(self, surrogate: SurrogateFitnessModel):
        self.surrogate = surrogate
    
    def expected_improvement(self,
                            genotype: Genotype,
                            best_fitness_so_far: float) -> float:
        """
        Expected improvement acquisition function
        
        Balances predicted fitness and uncertainty
        """
        pred_fitness, uncertainty = self.surrogate.predict_fitness(genotype)
        
        if uncertainty == 0:
            return 0.0
        
        # Improvement over current best
        improvement = pred_fitness - best_fitness_so_far
        
        # Expected improvement (assumes Gaussian uncertainty)
        z = improvement / uncertainty
        ei = improvement * norm.cdf(z) + uncertainty * norm.pdf(z)
        
        return ei
    
    def upper_confidence_bound(self,
                              genotype: Genotype,
                              beta: float = 2.0) -> float:
        """
        UCB acquisition function
        
        pred_fitness + beta * uncertainty
        """
        pred_fitness, uncertainty = self.surrogate.predict_fitness(genotype)
        return pred_fitness + beta * uncertainty


class SurrogateAssistedEvolution:
    """
    Evolution with surrogate-based fitness prediction
    
    Only evaluate promising genotypes fully
    """
    
    def __init__(self, population_size: int = 200):
        self.population_size = population_size
        self.surrogate = SurrogateFitnessModel()
        self.acquisition = AcquisitionFunction(self.surrogate)
        self.evaluation_budget = 1000  # Max full evaluations
        self.evaluations_used = 0
    
    def evolve_with_surrogate(self,
                             task_distribution: TaskDistribution,
                             num_generations: int = 100):
        """
        Surrogate-assisted evolutionary algorithm
        """
        population = initialize_population(self.population_size)
        
        # Evaluate initial population fully
        for genotype in population:
            fitness = self.full_evaluation(genotype, task_distribution)
            self.surrogate.add_evaluation(genotype, fitness)
            genotype.fitness = fitness
        
        for generation in range(num_generations):
            # Generate large candidate pool
            candidates = []
            for _ in range(self.population_size * 5):
                parent = random.choice(population)
                offspring = self.mutate(parent)
                candidates.append(offspring)
            
            # Predict fitness for all candidates
            predicted_fitnesses = []
            for candidate in candidates:
                pred_fit, _ = self.surrogate.predict_fitness(candidate)
                predicted_fitnesses.append(pred_fit)
            
            # Select top candidates by acquisition function
            acquisition_scores = [
                self.acquisition.upper_confidence_bound(c) 
                for c in candidates
            ]
            top_indices = np.argsort(acquisition_scores)[-self.population_size:]
            selected_candidates = [candidates[i] for i in top_indices]
            
            # Fully evaluate only top candidates (within budget)
            num_to_evaluate = min(
                len(selected_candidates),
                self.evaluation_budget - self.evaluations_used
            )
            
            for i in range(num_to_evaluate):
                candidate = selected_candidates[i]
                fitness = self.full_evaluation(candidate, task_distribution)
                self.surrogate.add_evaluation(candidate, fitness)
                candidate.fitness = fitness
                self.evaluations_used += 1
            
            # For remaining candidates, use surrogate predictions
            for i in range(num_to_evaluate, len(selected_candidates)):
                candidate = selected_candidates[i]
                pred_fit, _ = self.surrogate.predict_fitness(candidate)
                candidate.fitness = pred_fit
            
            # Update population
            population = selected_candidates
            
            # Log progress
            best_fitness = max(g.fitness for g in population)
            print(f"Gen {generation}: Best fitness = {best_fitness:.4f}, "
                  f"Evaluations used = {self.evaluations_used}/{self.evaluation_budget}")
            
            if self.evaluations_used >= self.evaluation_budget:
                print("Evaluation budget exhausted")
                break
        
        return population
    
    def full_evaluation(self, genotype: Genotype, task_distribution: TaskDistribution) -> float:
        """Expensive full evaluation"""
        phenotype = develop_phenotype_staged(genotype)
        
        total_fitness = 0.0
        for task in task_distribution.sample(num_tasks=10):
            learner = GradientLearner(phenotype)
            learner.learn_from_experience(task.get_training_data(), num_epochs=10)
            
            test_performance = evaluate_phenotype(phenotype, task.get_test_data())
            total_fitness += test_performance
        
        return total_fitness / 10
```

#### 8.1.2 Weight Inheritance and Transfer Learning

```python
class WeightInheritance:
    """
    Transfer learned weights between similar genotypes
    
    Avoids training from scratch after small mutations
    """
    
    def __init__(self):
        self.weight_library = {}  # Maps genotype fingerprints to trained weights
    
    def inherit_weights(self,
                       parent: Genotype,
                       parent_phenotype: Phenotype,
                       child: Genotype) -> Phenotype:
        """
        Initialize child's weights from trained parent
        
        Strategy:
        1. Identify corresponding modules between parent and child
        2. Copy weights for unchanged modules
        3. Initialize new modules using related parent modules
        """
        child_phenotype = develop_phenotype_staged(child)
        
        # Module correspondence
        correspondence = self.match_modules(parent, child)
        
        for child_module_id, parent_module_id in correspondence.items():
            if parent_module_id is not None:
                # Copy weights from parent
                child_module = child_phenotype.modules[child_module_id]
                parent_module = parent_phenotype.modules[parent_module_id]
                
                self.copy_compatible_weights(child_module, parent_module)
        
        return child_phenotype
    
    def match_modules(self,
                     parent: Genotype,
                     child: Genotype) -> Dict[str, Optional[str]]:
        """
        Find correspondence between parent and child modules
        
        Uses structural similarity and module IDs
        """
        correspondence = {}
        
        for child_module in child.modules:
            # Try exact ID match first
            parent_module = next((m for m in parent.modules if m.id == child_module.id), None)
            
            if parent_module is None:
                # Find most similar parent module
                similarities = [
                    self.module_similarity(child_module, pm)
                    for pm in parent.modules
                ]
                best_match_idx = np.argmax(similarities)
                
                if similarities[best_match_idx] > 0.5:  # Threshold
                    parent_module = parent.modules[best_match_idx]
            
            correspondence[child_module.id] = parent_module.id if parent_module else None
        
        return correspondence
    
    def copy_compatible_weights(self,
                               target_module: nn.Module,
                               source_module: nn.Module):
        """
        Copy weights between modules, handling dimension mismatches
        """
        source_state = source_module.state_dict()
        target_state = target_module.state_dict()
        
        for param_name in target_state:
            if param_name in source_state:
                source_param = source_state[param_name]
                target_param = target_state[param_name]
                
                if source_param.shape == target_param.shape:
                    # Exact match: copy directly
                    target_state[param_name] = source_param.clone()
                
                else:
                    # Dimension mismatch: copy compatible subsets
                    min_shape = tuple(min(s, t) for s, t in zip(source_param.shape, target_param.shape))
                    slices = tuple(slice(0, m) for m in min_shape)
                    
                    target_state[param_name][slices] = source_param[slices].clone()
        
        target_module.load_state_dict(target_state)
```

#### 8.1.3 Parallelization and Distributed Evolution

```python
class DistributedEvolution:
    """
    Parallelize evolution across multiple workers
    
    Each worker evaluates a subset of the population
    """
    
    def __init__(self, num_workers: int = 8):
        self.num_workers = num_workers
        self.pool = multiprocessing.Pool(num_workers)
    
    def parallel_evaluation(self,
                           population: List[Genotype],
                           task_distribution: TaskDistribution) -> np.ndarray:
        """
        Evaluate population in parallel
        
        Distributes genotypes across workers
        """
        # Create evaluation tasks
        tasks = [(genotype, task_distribution) for genotype in population]
        
        # Parallel map
        fitnesses = self.pool.starmap(self.evaluate_single, tasks)
        
        return np.array(fitnesses)
    
    def evaluate_single(self,
                       genotype: Genotype,
                       task_distribution: TaskDistribution) -> float:
        """Single genotype evaluation (executed by worker)"""
        phenotype = develop_phenotype_staged(genotype)
        
        total_fitness = 0.0
        for task in task_distribution.sample(num_tasks=5):
            learner = GradientLearner(phenotype)
            learner.learn_from_experience(task.get_training_data(), num_epochs=5)
            
            test_performance = evaluate_phenotype(phenotype, task.get_test_data())
            total_fitness += test_performance
        
        return total_fitness / 5


class AsynchronousEvolution:
    """
    Asynchronous evolution: don't wait for all evaluations to complete
    
    As soon as any evaluation finishes, use it to update population
    """
    
    def __init__(self):
        self.population = []
        self.evaluation_queue = []
        self.result_queue = []
    
    async def async_evolve(self,
                          initial_population: List[Genotype],
                          task_distribution: TaskDistribution,
                          num_evaluations: int = 10000):
        """
        Asynchronous evolutionary algorithm
        """
        self.population = initial_population
        
        # Submit initial evaluations
        for genotype in self.population:
            self.submit_evaluation(genotype, task_distribution)
        
        evaluations_completed = 0
        
        while evaluations_completed < num_evaluations:
            # Wait for any evaluation to complete
            genotype, fitness = await self.get_next_result()
            genotype.fitness = fitness
            evaluations_completed += 1
            
            # Update population immediately (replace worst individual)
            worst_idx = np.argmin([g.fitness for g in self.population])
            if fitness > self.population[worst_idx].fitness:
                self.population[worst_idx] = genotype
            
            # Generate new offspring and submit evaluation
            parent = self.tournament_select(self.population)
            offspring = self.mutate(parent)
            self.submit_evaluation(offspring, task_distribution)
            
            if evaluations_completed % 100 == 0:
                best_fitness = max(g.fitness for g in self.population)
                print(f"Evaluations: {evaluations_completed}, Best: {best_fitness:.4f}")
        
        return self.population
    
    async def submit_evaluation(self, genotype: Genotype, task_distribution: TaskDistribution):
        """Submit genotype for asynchronous evaluation"""
        # In practice, this would submit to a job queue
        task = asyncio.create_task(
            self.evaluate_async(genotype, task_distribution)
        )
        self.evaluation_queue.append(task)
    
    async def evaluate_async(self, genotype: Genotype, task_distribution: TaskDistribution):
        """Asynchronous evaluation (simulated)"""
        # Simulate evaluation time
        await asyncio.sleep(random.uniform(1, 5))
        
        fitness = self.evaluate_genotype(genotype, task_distribution)
        return genotype, fitness
```

### 8.2 Gradient-Guided Evolution

Hybridize discrete evolution with continuous gradient information:

```python
class GradientGuidedEvolution:
    """
    Use gradient information to guide evolutionary search
    
    For continuous hyperparameters, compute gradients of fitness w.r.t. parameters
    """
    
    def __init__(self):
        self.gradient_steps = 3  # Number of gradient steps per generation
        self.learning_rate = 0.01
    
    def evolve_with_gradients(self,
                             population: List[Genotype],
                             task_distribution: TaskDistribution) -> List[Genotype]:
        """
        Apply gradient-based updates to continuous hyperparameters
        """
        updated_population = []
        
        for genotype in population:
            # Make hyperparameters differentiable
            differentiable_genotype = self.make_differentiable(genotype)
            
            # Gradient descent on hyperparameters
            for step in range(self.gradient_steps):
                # Evaluate fitness
                phenotype = develop_phenotype_staged(differentiable_genotype)
                fitness = self.differentiable_fitness(phenotype, task_distribution)
                
                # Backpropagate through development and evaluation
                fitness.backward()
                
                # Update hyperparameters
                with torch.no_grad():
                    for module in differentiable_genotype.modules:
                        for param_name, param in module.hyperparams.items():
                            if isinstance(param, torch.Tensor) and param.grad is not None:
                                param.data += self.learning_rate * param.grad
                                param.grad.zero_()
            
            # Convert back to standard genotype
            updated_genotype = self.to_standard_genotype(differentiable_genotype)
            updated_population.append(updated_genotype)
        
        return updated_population
    
    def make_differentiable(self, genotype: Genotype) -> Genotype:
        """
        Convert continuous hyperparameters to torch.Tensor with requires_grad=True
        """
        diff_genotype = copy.deepcopy(genotype)
        
        for module in diff_genotype.modules:
            for param_name, param_value in module.hyperparams.items():
                if isinstance(param_value, (int, float)):
                    module.hyperparams[param_name] = torch.tensor(
                        float(param_value),
                        requires_grad=True
                    )
        
        return diff_genotype
    
    def differentiable_fitness(self,
                              phenotype: Phenotype,
                              task_distribution: TaskDistribution) -> torch.Tensor:
        """
        Compute fitness in a differentiable manner
        
        Fitness must be differentiable w.r.t. hyperparameters
        """
        total_fitness = torch.tensor(0.0, requires_grad=True)
        
        for task in task_distribution.sample(num_tasks=3):
            # Quick training
            learner = GradientLearner(phenotype, learning_rate=0.01)
            learner.learn_from_experience(task.get_training_data(), num_epochs=3)
            
            # Evaluate (must be differentiable)
            test_data = task.get_test_data()
            for batch in test_data:
                inputs, targets = batch
                outputs = phenotype.forward(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                # Fitness = negative loss
                total_fitness = total_fitness + (-loss)
        
        return total_fitness / 3


class DARTSStyleArchitectureSearch:
    """
    DARTS-style continuous relaxation of architecture search
    
    Represents architecture as continuous mixture of operations,
    then discretize after optimization
    """
    
    def __init__(self, operations: List[str]):
        self.operations = operations
        self.num_ops = len(operations)
    
    def create_mixed_operation(self, edge_id: str) -> nn.Module:
        """
        Mixed operation: weighted sum of all possible operations
        
        Weights (architecture parameters) are learned
        """
        return MixedOp(self.operations, edge_id)
    
    def search(self,
              task_data: DataLoader,
              num_epochs: int = 50):
        """
        Optimize architecture parameters via gradient descent
        """
        # Create super-network with mixed operations
        super_network = self.create_super_network()
        
        # Split data
        train_data, val_data = split_data(task_data, ratio=0.5)
        
        # Optimizers
        network_optimizer = torch.optim.SGD(
            super_network.weight_parameters(),
            lr=0.025,
            momentum=0.9
        )
        
        arch_optimizer = torch.optim.Adam(
            super_network.arch_parameters(),
            lr=3e-4
        )
        
        for epoch in range(num_epochs):
            # Update network weights on train data
            for batch in train_data:
                inputs, targets = batch
                outputs = super_network(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                network_optimizer.zero_grad()
                loss.backward()
                network_optimizer.step()
            
            # Update architecture parameters on validation data
            for batch in val_data:
                inputs, targets = batch
                outputs = super_network(inputs)
                loss = F.cross_entropy(outputs, targets)
                
                arch_optimizer.zero_grad()
                loss.backward()
                arch_optimizer.step()
        
        # Discretize architecture
        final_genotype = super_network.derive_genotype()
        
        return final_genotype


class MixedOp(nn.Module):
    """
    Continuous mixture of operations
    
    output = sum_i(softmax(alpha)_i * op_i(input))
    """
    
    def __init__(self, operations: List[str], edge_id: str):
        super().__init__()
        self.operations = nn.ModuleList([
            self.create_operation(op_name)
            for op_name in operations
        ])
        self.alpha = nn.Parameter(torch.randn(len(operations)))
        self.edge_id = edge_id
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Compute weighted sum of all operations"""
        weights = F.softmax(self.alpha, dim=0)
        
        output = sum(w * op(x) for w, op in zip(weights, self.operations))
        return output
    
    def derive_operation(self) -> str:
        """Select single operation with highest weight"""
        weights = F.softmax(self.alpha, dim=0)
        best_op_idx = torch.argmax(weights).item()
        return self.operations[best_op_idx]
```

### 8.3 Checkpointing and Experiment Management

```python
class EvolutionManager:
    """
    Manage long-running evolutionary experiments
    
    - Checkpointing
    - Logging
    - Resumption from failures
    - Analysis and visualization
    """
    
    def __init__(self, experiment_name: str, checkpoint_dir: str = "./checkpoints"):
        self.experiment_name = experiment_name
        self.checkpoint_dir = Path(checkpoint_dir) / experiment_name
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.history = {
            'generation': [],
            'best_fitness': [],
            'mean_fitness': [],
            'diversity': [],
            'population_snapshots': []
        }
    
    def save_checkpoint(self,
                       generation: int,
                       population: List[Genotype],
                       additional_state: Dict = None):
        """Save complete state for resumption"""
        checkpoint = {
            'generation': generation,
            'population': [self.serialize_genotype(g) for g in population],
            'history': self.history,
            'additional_state': additional_state or {}
        }
        
        checkpoint_path = self.checkpoint_dir / f"gen_{generation:06d}.pkl"
        with open(checkpoint_path, 'wb') as f:
            pickle.dump(checkpoint, f)
        
        # Keep only recent checkpoints (save space)
        self.cleanup_old_checkpoints(keep_every=50)
    
    def load_checkpoint(self, generation: int = None) -> Tuple[int, List[Genotype], Dict]:
        """
        Load checkpoint to resume evolution
        
        If generation=None, loads most recent checkpoint
        """
        if generation is None:
            # Find most recent
            checkpoints = list(self.checkpoint_dir.glob("gen_*.pkl"))
            if not checkpoints:
                raise ValueError("No checkpoints found")
            checkpoint_path = max(checkpoints, key=lambda p: p.stat().st_mtime)
        else:
            checkpoint_path = self.checkpoint_dir / f"gen_{generation:06d}.pkl"
        
        with open(checkpoint_path, 'rb') as f:
            checkpoint = pickle.load(f)
        
        generation = checkpoint['generation']
        population = [self.deserialize_genotype(g) for g in checkpoint['population']]
        self.history = checkpoint['history']
        additional_state = checkpoint.get('additional_state', {})
        
        return generation, population, additional_state
    
    def log_generation(self,
                      generation: int,
                      population: List[Genotype],
                      fitnesses: np.ndarray):
        """Log statistics for this generation"""
        self.history['generation'].append(generation)
        self.history['best_fitness'].append(np.max(fitnesses))
        self.history['mean_fitness'].append(np.mean(fitnesses))
        
        # Diversity
        diversity = self.compute_diversity(population)
        self.history['diversity'].append(diversity)
        
        # Save population snapshot periodically
        if generation % 10 == 0:
            self.history['population_snapshots'].append({
                'generation': generation,
                'population': copy.deepcopy(population),
                'fitnesses': fitnesses.copy()
            })
        
        # Save logs
        self.save_logs()
    
    def save_logs(self):
        """Save history to JSON"""
        log_path = self.checkpoint_dir / "evolution_log.json"
        
        # Convert to JSON-serializable format
        serializable_history = {
            'generation': self.history['generation'],
            'best_fitness': [float(f) for f in self.history['best_fitness']],
            'mean_fitness': [float(f) for f in self.history['mean_fitness']],
            'diversity': [float(d) for d in self.history['diversity']]
        }
        
        with open(log_path, 'w') as f:
            json.dump(serializable_history, f, indent=2)
    
    def visualize_progress(self):
        """Create visualization of evolutionary progress"""
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Fitness over time
        axes[0, 0].plot(self.history['generation'], self.history['best_fitness'], label='Best')
        axes[0, 0].plot(self.history['generation'], self.history['mean_fitness'], label='Mean')
        axes[0, 0].set_xlabel('Generation')
        axes[0, 0].set_ylabel('Fitness')
        axes[0, 0].set_title('Fitness Evolution')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Diversity over time
        axes[0, 1].plot(self.history['generation'], self.history['diversity'])
        axes[0, 1].set_xlabel('Generation')
        axes[0, 1].set_ylabel('Population Diversity')
        axes[0, 1].set_title('Genetic Diversity')
        axes[0, 1].grid(True)


