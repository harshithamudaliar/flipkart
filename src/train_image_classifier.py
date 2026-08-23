import torch
import torchvision
import torch.nn as nn
import sklearn.model_selection 

#transform the FashionMNIST dataset to be compatible with ResNet18
transform = torchvision.transforms.Compose([
    torchvision.transforms.Grayscale(num_output_channels=3),
    torchvision.transforms.Resize((224, 224)),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225])
])
train_dataset = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.FashionMNIST(root='./data', train=False, download=True, transform=transform)


#stratified split of train into train and val subsets
splits = sklearn.model_selection.StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_indices, val_indices = next(splits.split(train_dataset.data, train_dataset.targets.numpy()))
train_subset = torch.utils.data.Subset(train_dataset, train_indices)
val_subset = torch.utils.data.Subset(train_dataset, val_indices)
print("Train dataset size:", len(train_subset), "| Validation dataset size:", len(val_subset), "| Test dataset size:", len(test_dataset))

#transfer learning model
model = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1
)
#freeze backbone
for param in model.parameters():
    param.requires_grad = False

#New classifier head
num_features = model.classifier[1].in_features 
model.eval()

# Extract and cache Backbone features before training classifier head
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

#extract and cache backbone features before classifier head
def extract_features(dataset, model, batch_size=64):
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=False)
    original_classifier = model.classifier
    model.classifier = nn.Identity()

    all_features = []
    all_labels = []
    with torch.no_grad():
        for i, (images, labels) in enumerate(loader):
            images = images.to(device)
            features = model(images)
            all_features.append(features.cpu())
            all_labels.append(labels)
            if i % 20 == 0:
                print(f"  batch {i}/{len(loader)}")

    model.classifier = original_classifier
    return torch.cat(all_features), torch.cat(all_labels)

print("Extracting train features...")
train_features, train_labels = extract_features(train_subset, model)
print("Extracting val features...")
val_features, val_labels = extract_features(val_subset, model)
print("Extracting test features...")
test_features, test_labels = extract_features(test_dataset, model)

print("Cached feature shapes:", train_features.shape, val_features.shape, test_features.shape)

# --- Train only the head on cached features ---
head = nn.Sequential(
    nn.Linear(num_features, 128),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(128, 10)
)

train_feat_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
val_feat_dataset = torch.utils.data.TensorDataset(val_features, val_labels)

BATCH_SIZE = 64
train_feat_loader = torch.utils.data.DataLoader(train_feat_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_feat_loader = torch.utils.data.DataLoader(val_feat_dataset, batch_size=BATCH_SIZE, shuffle=False)

optimizer = torch.optim.Adam(head.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()

EPOCHS = 10
for epoch in range(EPOCHS):
    head.train()
    for features, labels in train_feat_loader:
        optimizer.zero_grad()
        outputs = head(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

    head.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for features, labels in val_feat_loader:
            outputs = head(features)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
    print(f"Epoch {epoch+1}/{EPOCHS} — val accuracy: {correct/total:.4f}")

import os

os.makedirs('models', exist_ok=True)
os.makedirs('data/cached_features', exist_ok=True)

torch.save(head.state_dict(), 'models/head_state.pt')

torch.save({'features': test_features, 'labels': test_labels}, 'data/cached_features/test.pt')

print("Saved head weights to models/head_state.pt")
print("Saved test features to data/cached_features/test.pt")
print("num_features (for rebuilding head in notebook):", num_features)