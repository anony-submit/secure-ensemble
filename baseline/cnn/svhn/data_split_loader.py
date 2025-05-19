import random
from torchvision import datasets, transforms
from collections import defaultdict
import torch

def create_balanced_data_loaders(num_users, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    data_len = len(full_dataset) // num_users
    user_loaders = []
    for i in range(num_users):
        start = i * data_len
        end = len(full_dataset) if i == num_users - 1 else (i + 1) * data_len
        subset = torch.utils.data.Subset(full_dataset, list(range(start, end)))
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        user_loaders.append(loader)
    return user_loaders

def create_custom_data_loaders(distribution_path, num_users, batch_size=32):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    full_dataset = datasets.SVHN(root='./data', split='train', download=True, transform=transform)
    class_indices = defaultdict(list)
    for idx, (img, label) in enumerate(full_dataset):
        class_indices[int(label)].append(idx)

    splits = defaultdict(list)
    current_split = -1
    with open(distribution_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "distribution" in line.lower():
                continue
            if line.startswith("Split"):
                current_split = int(line.split()[1].strip(":")) - 1
            elif line.startswith("Class"):
                class_id, count = map(int, line.replace("Class", "").split(":"))
                splits[current_split].append((class_id, count))
    user_loaders = []
    for user_id in range(num_users):
        user_indices = []
        for class_id, count in splits[user_id]:
            available = class_indices[class_id]
            if len(available) < count:
                count = len(available)
            samples = random.sample(available, count)
            user_indices.extend(samples)
            class_indices[class_id] = [i for i in available if i not in samples]
        subset = torch.utils.data.Subset(full_dataset, user_indices)
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        user_loaders.append(loader)
    return user_loaders
