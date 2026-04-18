# Evolutionary Neural Systems Architecture

## Core Components
- **Genotype**: The compact specification of a neural architecture.
- **Phenotype**: The executable PyTorch module instantiated from a Genotype.
- **Modules**: Pluggable components (Linear, Attention, RNN, CNN).
- **Evolution Engine**: Manages the lifecycle of populations, selection, mutation, and crossover.

## Directory Structure
- `src/core`: Evolution logic, Phenotype mapping, Genotype definitions.
- `src/modules`: Individual neural components.
- `src/utils`: Supporting scripts for logging and configuration.
- `experiments`: Scripts to test models on ARC, HLE, and other datasets.

## Development Principles
1. **Modularity**: New neural component types should be easy to add.
2. **Reproducibility**: Evolution loops must log configurations and preserve histories.
3. **Efficiency**: Phenotype instantiations should minimize overhead during evaluations.
