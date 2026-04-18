import unittest
import torch
from src.modules.linear import LinearModule
from src.core.genotype import ModuleGene

class TestLinearModule(unittest.TestCase):
    def test_forward(self):
        gene = ModuleGene("M1", "linear", {"dim": 32})
        mod = LinearModule(gene, 16)
        x = torch.randn(2, 16)
        out = mod(x)
        self.assertEqual(out.shape, (2, 32))

if __name__ == "__main__":
    unittest.main()
