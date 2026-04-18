#!/bin/bash
# Wrapper script to run evolution
echo "Starting Evolutionary Run..."
python -c "
from src.core.evolution import EvolutionarySystem
from src.core.genotype import Genotype, ModuleGene
import random

def get_init():
    g = Genotype()
    g.add_module(ModuleGene('M0', 'linear', {'dim': 64}))
    return g

sys = EvolutionarySystem(10, 0.1)
sys.initialize_population(get_init)

def dummy_eval(g):
    return {'accuracy': random.random()}

for _ in range(3):
    sys.evolve_generation(dummy_eval)
print('Evolution finished.')
"
