"""
Script to visualize genotype architectures.
"""
def visualize(genotype):
    print("Visualizing genotype structure...")
    for mod in genotype.modules:
        print(f"Module: {mod.id} - {mod.type}")
    for conn in genotype.connections:
        print(f"Connection: {conn.source} -> {conn.target}")

if __name__ == "__main__":
    print("Graph visualization tool ready.")
