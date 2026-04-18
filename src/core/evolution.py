"""
Evolutionary System loop for managing population and evolution logic.
"""
import copy
import random
import numpy as np

class EvolutionarySystem:
    def __init__(self, population_size=50, mutation_rate=0.2):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.population = []
        self.generation = 0
        self.best_fitness_history = []

    def initialize_population(self, initial_genotype_generator):
        for _ in range(self.population_size):
            self.population.append(initial_genotype_generator())

    def evolve_generation(self, evaluate_fn):
        fitness_scores = {}
        for genotype in self.population:
            scores = evaluate_fn(genotype)
            fitness = scores.get('accuracy', 0.0) - scores.get('complexity', 0) * 0.001
            fitness_scores[id(genotype)] = fitness
            genotype.fitness_history.append(fitness)

        sorted_pop = sorted(self.population, key=lambda g: fitness_scores[id(g)], reverse=True)
        parents = sorted_pop[:self.population_size // 2]

        offspring = []
        while len(offspring) < self.population_size:
            p1, p2 = random.sample(parents, 2)
            child = self._crossover(p1, p2)
            child.mutate(self.mutation_rate)
            offspring.append(child)

        self.population = offspring
        self.generation += 1
        return fitness_scores[id(sorted_pop[0])]

    def _crossover(self, p1, p2):
        # Simplified crossover implementation
        child = copy.deepcopy(p1)
        return child
