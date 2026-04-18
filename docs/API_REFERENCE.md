# API Reference

## Genotype API
`Genotype.add_module(ModuleGene)`: Adds a module specification.
`Genotype.add_connection(ConnectionGene)`: Adds a directed link between modules.
`Genotype.mutate(rate)`: Applies random valid mutations.

## Phenotype API
`Phenotype(genotype, input_dim, output_dim, module_factory)`: Creates a PyTorch `nn.Module`.
`Phenotype.forward(x)`: Executes the full forward pass based on the architectural graph.

## Evolution API
`EvolutionarySystem(population_size, mutation_rate)`: Initializes the engine.
`EvolutionarySystem.evolve_generation(evaluate_fn)`: Steps the population forward by one generation based on fitness values from `evaluate_fn`.
