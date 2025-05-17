import torch
from torchvision import models, datasets, transforms
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

'''
1. Model Architecture Verification
Ensure that the final layer of the model correctly outputs two classes for binary classification.
'''
def test_model_output():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = torch.nn.Linear(num_ftrs, 2)  # Binary classification
    model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    output = model(dummy_input)
    assert output.shape == (1, 2), f"Expected output shape (1, 2), got {output.shape}"


'''
2. DataLoader Integrity Check
Verify that the DataLoader correctly processes and batches the data.
'''


def test_dataloader():
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    dataset = datasets.ImageFolder('/Users/youngbinpyo/Documents/KTH/DL/Project/DD2424-Project/data/train', transform=transform) # change this to your directory
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)
    images, labels = next(iter(dataloader))
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)



'''
3. Training Loop Functionality
Confirm that a single training iteration runs without errors.
'''

def test_training_step():
    model = models.resnet18(pretrained=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, 2)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model.train()
    inputs = torch.randn(4, 3, 224, 224)
    labels = torch.tensor([0, 1, 0, 1])
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()
    print(f"Training step completed with loss: {loss.item()}")


'''
4. Evaluation Metrics Calculation
Ensure that evaluation metrics like accuracy are computed correctly.
'''

def test_evaluation_metrics():
    outputs = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    labels = torch.tensor([0, 1])
    _, preds = torch.max(outputs, 1)
    accuracy = torch.sum(preds == labels).item() / len(labels)
    assert accuracy == 1.0, f"Expected accuracy 1.0, got {accuracy}"

