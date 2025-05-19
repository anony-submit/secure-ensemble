import torch
import torchvision.transforms as transforms
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

class StandardAugment:
    def __init__(self):
        self.transforms_list = [
            ('rotation', transforms.RandomRotation(degrees=15)),
            ('affine', transforms.RandomAffine(
                degrees=0,
                translate=(0.1, 0.1),
                scale=(0.9, 1.1),
                shear=(-5, 5)
            )),
            ('crop', transforms.RandomCrop(32, padding=4)),
            ('color', transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            )),
            ('blur', transforms.GaussianBlur(3, sigma=(0.1, 0.2)))
        ]
    
    def __call__(self, img):
        n_transforms = 3
        selected_indices = np.random.choice(len(self.transforms_list), n_transforms, replace=False)
        selected_transforms = [self.transforms_list[i][1] for i in selected_indices]
        
        for transform in selected_transforms:
            img = transform(img)
        return img

class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, original_dataset, augmentation_type='standard'):
        self.original_dataset = original_dataset
        self.augmentation_type = augmentation_type
        
        if augmentation_type == 'standard':
            self.transform = StandardAugment()
        elif augmentation_type == 'randaug':
            self.transform = transforms.RandAugment(num_ops=2, magnitude=4)
        else:
            self.transform = None

    def __getitem__(self, idx):
        img, label = self.original_dataset[idx]
        
        if isinstance(img, torch.Tensor):
            mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
            img = img * std + mean
            img = transforms.ToPILImage()(img)

        if self.transform:
            img = self.transform(img)
            
        img = transforms.ToTensor()(img)
        img = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(img)
        return img, label

    def __len__(self):
        return len(self.original_dataset)

def augment_subset(subset, augmentation_type='standard', n=1, apply_augment=True):
    if not apply_augment:
        return subset
        
    if n == 1:
        return subset

    aug_ratio = n if n <= 5 else 10
    aug_ratio = aug_ratio - 1

    aug_datasets = [AugmentedDataset(subset, augmentation_type) for _ in range(aug_ratio)]
    combined_dataset = torch.utils.data.ConcatDataset([subset] + aug_datasets)
    
    return combined_dataset

def create_dirichlet_split(dataset, n_splits, alpha, save_path=None, seed=42):
    np.random.seed(seed)
    n_classes = 10
    class_indices = {i: [] for i in range(n_classes)}
    
    for idx, (_, label) in enumerate(dataset):
        class_indices[label].append(idx)
    
    proportions = np.random.dirichlet(np.repeat(alpha, n_splits), size=n_classes)
    
    split_indices = [[] for _ in range(n_splits)]
    distribution = np.zeros((n_splits, n_classes))
    
    for class_idx, indices in class_indices.items():
        np.random.shuffle(indices)
        n_samples = len(indices)
        
        split_sizes = (proportions[class_idx] * n_samples).astype(int)
        split_sizes[-1] = n_samples - split_sizes[:-1].sum()
        
        start_idx = 0
        for split_idx, size in enumerate(split_sizes):
            split_indices[split_idx].extend(indices[start_idx:start_idx + size])
            distribution[split_idx][class_idx] = size
            start_idx += size
    
    if save_path:
        plt.figure(figsize=(10, 8))
        sns.heatmap(distribution, annot=True, fmt='g', cmap='YlOrRd')
        plt.title(f'Class Distribution (α={alpha})')
        plt.xlabel('Class')
        plt.ylabel('Split')
        plt.savefig(save_path)
        plt.close()
    
    return [torch.utils.data.Subset(dataset, indices) for indices in split_indices], distribution