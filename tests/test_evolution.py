import unittest
from src.core.evolution import EvolutionarySystem
from src.core.genotype import Genotype

class TestEvolutionSystem(unittest.TestCase):
    def test_init(self):
        sys = EvolutionarySystem(10, 0.2)
        sys.initialize_population(Genotype)
        self.assertEqual(len(sys.population), 10)

if __name__ == "__main__":
    unittest.main()
