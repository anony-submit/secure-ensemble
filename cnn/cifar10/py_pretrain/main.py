import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from datetime import datetime
from sklearn.metrics import accuracy_score
from models import TinyCNN
from data_utils import create_dirichlet_split, augment_subset
from train_utils import train_model, ensemble_predict
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n🖥 Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)

    test_size = len(testset)
    fixed_indices = torch.randperm(test_size)[:100]
    fixed_testset = torch.utils.data.Subset(testset, fixed_indices)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=4)
    fixed_testloader = torch.utils.data.DataLoader(fixed_testset, batch_size=128, shuffle=False, num_workers=4)

    test_loader = torch.utils.data.DataLoader(fixed_testset, batch_size=100)
    test_data = next(iter(test_loader))[0]
    test_labels = np.array([testset.targets[i] for i in fixed_indices])
    test_data = test_data.numpy().reshape(100, -1)
    os.makedirs('test_data', exist_ok=True)
    df = pd.DataFrame(test_data)
    df['label'] = test_labels
    df.to_csv('test_data/cifar10_test.csv', index=False)

    base_dir = 'results'
    experiment_dir = os.path.join(base_dir, f'cifar10_experiment_{datetime.now().strftime("%Y%m%d_%H%M%S")}')
    os.makedirs(experiment_dir, exist_ok=True)

    n = 10
    alpha = 0.1
    print(f"\n📊 Starting experiment with n = {n}, Dirichlet α = {alpha}")

    subsets, distribution = create_dirichlet_split(trainset, n, alpha, save_path=os.path.join(experiment_dir, 'distribution.png'))

    with open(os.path.join(experiment_dir, 'distribution.txt'), 'w') as f:
        for i in range(n):
            f.write(f"Split {i+1} class distribution:\n")
            for j in range(10):
                f.write(f"  Class {j}: {int(distribution[i][j])}\n")
            f.write("\n")

    loss_types = ['vanilla', 'balsoftmax'] + [f'balsoftmax_entropy_{int(w*10)}' for w in [0.1, 0.5, 1.0, 1.5, 2.0, 3.0]]

    for loss_type in loss_types:
        print(f"\n🔵 Training TinyCNN ensemble with loss: {loss_type}")
        loss_dir = os.path.join(experiment_dir, 'cnn', loss_type)
        os.makedirs(loss_dir, exist_ok=True)

        models = []
        accuracies = []
        fixed_accuracies = []

        for i, subset in enumerate(subsets):
            print(f"  🛠 Training model {i+1}/{n}")
            if loss_type == 'vanilla':
                aug_subset = augment_subset(subset, augmentation_type='randaug', n=2)
                trainloader = torch.utils.data.DataLoader(aug_subset, batch_size=128, shuffle=True, num_workers=4)
                sample_per_class = None
            else:
                trainloader = torch.utils.data.DataLoader(subset, batch_size=128, shuffle=True, num_workers=4)
                sample_per_class = np.bincount([trainset.targets[idx] for idx in subset.indices], minlength=10)

            model = TinyCNN().to(device)
            acc, fixed_acc, params = train_model(
                model, trainloader, testloader, fixed_testloader, device,
                model_save_path=os.path.join(loss_dir, f'model_params{i+1}.json'),
                training_mode=loss_type, sample_per_class=sample_per_class
            )
            models.append(model)
            accuracies.append(acc)
            fixed_accuracies.append(fixed_acc)

        ensemble_preds = ensemble_predict(models, testloader, device)
        ensemble_fixed_preds = ensemble_predict(models, fixed_testloader, device)
        ensemble_acc = accuracy_score(testset.targets, ensemble_preds) * 100
        ensemble_fixed_acc = accuracy_score([testset.targets[i] for i in fixed_indices], ensemble_fixed_preds) * 100

        with open(os.path.join(loss_dir, 'results.txt'), 'w') as f:
            f.write("Individual Model Accuracies:\n")
            for i, (acc, fixed_acc) in enumerate(zip(accuracies, fixed_accuracies)):
                f.write(f"Model {i+1}: Test Acc = {acc:.2f}%, Fixed Test Acc = {fixed_acc:.2f}%\n")
            f.write(f"\nMean Test Accuracy: {np.mean(accuracies):.2f}%\n")
            f.write(f"Mean Fixed Test Accuracy: {np.mean(fixed_accuracies):.2f}%\n")
            f.write(f"Ensemble Test Accuracy: {ensemble_acc:.2f}%\n")
            f.write(f"Ensemble Fixed Test Accuracy: {ensemble_fixed_acc:.2f}%\n")


if __name__ == '__main__':
    main()