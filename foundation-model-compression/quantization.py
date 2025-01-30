"""
**QUANTIZATION:**
* Quantization reduces the precision of the model's weights and activations (e.g., from 32-bit floating point to 8-bit integers). This makes the model smaller and faster, which is useful for deployment on resource-constrained devices like mobile phones or edge devices.
* Deploying LLMs on mobile devices or embedded systems where memory and compute resources are limited.
"""

import torch
from torch.quantization import quantize_dynamic

# Load a pre-trained model
model = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=True)

# Print the original model size
original_size = sum(p.numel() * p.element_size() for p in model.parameters())
print(f"Original model size: {original_size} bytes")

# Quantize the model dynamically (post-training quantization)
# We quantize only the linear and convolutional layers
quantized_model = quantize_dynamic(model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8)

# Print the quantized model size
quantized_size = sum(p.numel() * p.element_size() for p in quantized_model.parameters())
print(f"Quantized model size: {quantized_size} bytes")

# Check the size reduction
print(f"Size reduction: {original_size - quantized_size} bytes")