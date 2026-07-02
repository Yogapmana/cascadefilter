import random
import sys

def generate_dimensions(num_matrices, base_size=100):
    dims = [random.randint(base_size // 2, base_size * 2) for _ in range(num_matrices + 1)]
    return dims

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <num_matrices>")
        sys.exit(1)
        
    num_matrices = int(sys.argv[1])
    dims = generate_dimensions(num_matrices)
    print(" ".join(map(str, dims)))
