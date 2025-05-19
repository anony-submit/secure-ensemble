import json
import random
from torchvision import datasets, transforms
from collections import defaultdict
import torch
import os

def parse_distribution_file(path):
    from collections import defaultdict
    splits = defaultdict(list)
    current_split = -1
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line or "distribution" in line.lower():
                continue
            if line.startswith("Split"):
                current_split = int(line.split()[1].strip(":")) - 1
            elif line.startswith("Class"):
                try:
                    class_id, count = map(int, line.replace("Class", "").split(":"))
                    splits[current_split].append((class_id, count))
                except Exception as e:
                    print(f"[ERROR] Could not parse line: '{line}' - {e}")
    return splits

def create_custom_data_loaders(distribution_path, num_users, batch_size=32):
    dist = parse_distribution_file(distribution_path)
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

    class_indices = defaultdict(list)
    for idx, (img, label) in enumerate(full_dataset):
        class_indices[label].append(idx)

    loaders = []
    for user_id in range(num_users):
        user_indices = []
        for class_id, count in dist[user_id]:
            samples = random.sample(class_indices[class_id], count)
            user_indices.extend(samples)
            # remove to avoid reuse
            class_indices[class_id] = [i for i in class_indices[class_id] if i not in samples]
        user_subset = torch.utils.data.Subset(full_dataset, user_indices)
        loader = torch.utils.data.DataLoader(user_subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders

def create_balanced_data_loaders(num_users, batch_size=32):
    transform = transforms.Compose([transforms.ToTensor()])
    full_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    data_len = len(full_dataset) // num_users
    loaders = []
    for i in range(num_users):
        start = i * data_len
        end = len(full_dataset) if i == num_users - 1 else (i + 1) * data_len
        subset = torch.utils.data.Subset(full_dataset, list(range(start, end)))
        loader = torch.utils.data.DataLoader(subset, batch_size=batch_size, shuffle=True)
        loaders.append(loader)
    return loaders
