"""
**PRUNNING:**
* Pruning removes less important weights or neurons from a model, reducing its size and improving inference speed without significantly affecting accuracy.
* Optimizing LLMs for deployment in environments with strict latency or memory constraints.
"""

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize model
model = SimpleModel()

# Prune 50% of the weights in the fully connected layer
prune.l1_unstructured(model.fc, name='weight', amount=0.5)

# Check pruned weights
print(model.fc.weight)

# Remove pruning reparameterization (optional)
prune.remove(model.fc, 'weight')