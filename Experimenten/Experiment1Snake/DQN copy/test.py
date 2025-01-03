import torch
import torch_directml

device = torch_directml.device()
print(f"Using device: {device}")

# Test tensor
x = torch.ones((1000, 1000), device=device)
y = torch.ones((1000, 1000), device=device)
z = x + y
print("Tensor calculation successful:", z[0, 0].item())  # Should print 2.0
