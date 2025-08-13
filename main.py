""" def main():
    print("Hello from triton-practice!")


if __name__ == "__main__":
    main()
 """
 
import torch
import torch.nn.functional as F

# Input values
x = torch.tensor([-1.0, 0.0, 1.6, 0.8, 2.1])

# Compute GELU
gelu_values = F.gelu(x, approximate="tanh")

# Print results
for i, val in enumerate(x):
    print(f"GELU({val.item()}) = {gelu_values[i].item()}")
