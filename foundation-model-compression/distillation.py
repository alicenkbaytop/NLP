"""
**DISTILLATION:**
* Distillation involves training a smaller "student" model to mimic the behavior of a larger "teacher" model. The student learns from the teacher's outputs (logits) or intermediate representations, achieving similar performance with fewer parameters.
* Creating smaller, faster models for real-time applications like chatbots or voice assistants.
"""

import torch
import torch.nn as nn
import torch.optim as optim

# Define a simple teacher and student model
class TeacherModel(nn.Module):
    def __init__(self):
        super(TeacherModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

class StudentModel(nn.Module):
    def __init__(self):
        super(StudentModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)

# Initialize models
teacher = TeacherModel()
student = StudentModel()

# Loss function (distillation loss)
criterion = nn.MSELoss()
optimizer = optim.Adam(student.parameters(), lr=0.001)

# Dummy data
inputs = torch.randn(100, 10)

# Training loop
for epoch in range(10):
    optimizer.zero_grad()

    # Forward pass through the teacher model
    with torch.no_grad():  # No need to compute gradients for the teacher
        teacher_outputs = teacher(inputs)

    # Forward pass through the student model
    student_outputs = student(inputs)

    # Compute the loss
    loss = criterion(student_outputs, teacher_outputs)

    # Backward pass and optimization
    loss.backward()
    optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item()}")