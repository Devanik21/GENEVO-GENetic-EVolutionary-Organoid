import unittest
from src.core.genotype import Genotype, ModuleGene, ConnectionGene

class TestGenotype(unittest.TestCase):
    def test_add_module(self):
        g = Genotype()
        g.add_module(ModuleGene("M0", "linear", {"dim": 128}))
        self.assertEqual(len(g.modules), 1)
        self.assertEqual(g.modules[0].id, "M0")

    def test_mutate(self):
        g = Genotype()
        g.add_module(ModuleGene("M0", "linear", {"dim": 128}))
        # High mutation rate to force mutation
        muts = g.mutate(mutation_rate=1.0)
        self.assertTrue(len(muts) > 0)

if __name__ == "__main__":
    unittest.main()
