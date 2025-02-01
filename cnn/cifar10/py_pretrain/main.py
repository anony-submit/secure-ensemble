import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from sklearn.metrics import accuracy_score
from datetime import datetime
import pandas as pd
from models import TinyCNN, ResNet18
from data_utils import augment_subset, create_dirichlet_split
from train_utils import train_model, ensemble_predict
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                         download=True, transform=transform)
    
    test_size = len(testset)
    fixed_indices = torch.randperm(test_size)[:100]
    fixed_testset = torch.utils.data.Subset(testset, fixed_indices)
    
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

    n_values = [1,2,5,10,20]
    batch_size = 128
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                          shuffle=False, num_workers=16)
    fixed_testloader = torch.utils.data.DataLoader(fixed_testset, batch_size=batch_size,
                                                shuffle=False, num_workers=16)

    for n in n_values:
        print(f"\n=== Starting experiment with n={n} ===")
        exp_dir = os.path.join(experiment_dir, f'n_{n}')
        os.makedirs(exp_dir, exist_ok=True)
        
        if n == 1:
            os.makedirs(os.path.join(exp_dir, 'resnet'), exist_ok=True)
            os.makedirs(os.path.join(exp_dir, 'cnn'), exist_ok=True)
            
            randaug_trainset = augment_subset(trainset, augmentation_type='randaug', n=2, apply_augment=True)
            trainloader = torch.utils.data.DataLoader(randaug_trainset, batch_size=batch_size,
                                                  shuffle=True, num_workers=16)
            
            print("\nTraining ResNet...")
            resnet = ResNet18().to(device)
            accuracy, fixed_accuracy, _ = train_model(resnet, trainloader, testloader, 
                                                  fixed_testloader, device)
            
            with open(os.path.join(exp_dir, 'resnet', 'results.txt'), 'w') as f:
                f.write(f"Test Accuracy: {accuracy:.2f}%\n")
                f.write(f"Fixed Test Accuracy: {fixed_accuracy:.2f}%\n")
            print(f"ResNet - Test Accuracy: {accuracy:.2f}%")
            print(f"ResNet - Fixed Test Accuracy: {fixed_accuracy:.2f}%")
            
            print("\nTraining CNN...")
            cnn = TinyCNN().to(device)
            accuracy, fixed_accuracy, params = train_model(cnn, trainloader, testloader, 
                                                      fixed_testloader, device,
                                                      os.path.join(exp_dir, 'cnn', 'model_params.json'), 
                                                      'vanilla')
            
            with open(os.path.join(exp_dir, 'cnn', 'results.txt'), 'w') as f:
                f.write(f"Test Accuracy: {accuracy:.2f}%\n")
                f.write(f"Fixed Test Accuracy: {fixed_accuracy:.2f}%\n")
            print(f"CNN - Test Accuracy: {accuracy:.2f}%")
            print(f"CNN - Fixed Test Accuracy: {fixed_accuracy:.2f}%")
        
        else:
            split_methods = [
                ('balanced', None),
                ('dirichlet0.1', 0.1),
                ('dirichlet0.5', 0.5)
            ]
            
            for method_name, alpha in split_methods:
                print(f"\nStarting {method_name} experiment...")
                method_dir = os.path.join(exp_dir, method_name)
                os.makedirs(method_dir, exist_ok=True)
                os.makedirs(os.path.join(method_dir, 'resnet'), exist_ok=True)
                os.makedirs(os.path.join(method_dir, 'cnn'), exist_ok=True)
                
                if alpha is None:
                    indices = torch.randperm(len(trainset))
                    split_size = len(trainset) // n
                    subsets = [torch.utils.data.Subset(trainset, 
                        indices[i*split_size:(i+1)*split_size]) for i in range(n)]
                    
                    distribution = np.zeros((n, 10))
                    for i, subset in enumerate(subsets):
                        distribution[i] = np.bincount([trainset.targets[idx] for idx in subset.indices], minlength=10)
                    
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(distribution, annot=True, fmt='g', cmap='YlOrRd')
                    plt.title(f'Class Distribution (Balanced)')
                    plt.xlabel('Class')
                    plt.ylabel('Split')
                    plt.savefig(os.path.join(method_dir, 'distribution.png'))
                    plt.close()
                else:
                    subsets, distribution = create_dirichlet_split(trainset, n, alpha,
                                                               save_path=os.path.join(method_dir, 'distribution.png'))
                
                with open(os.path.join(method_dir, 'distribution.txt'), 'w') as f:
                    f.write(f"Dataset distribution for {method_name}:\n\n")
                    for i in range(n):
                        f.write(f"Split {i+1}:\n")
                        for j in range(10):
                            f.write(f"Class {j}: {int(distribution[i][j])}\n")
                        f.write("\n")

                print("\nTraining ResNet ensemble...")
                models = []
                accuracies = []
                fixed_accuracies = []
                
                for i, subset in enumerate(subsets):
                    print(f"\nTraining ResNet {i+1}/{n}")
                    randaug_trainset = augment_subset(subset, augmentation_type='randaug', n=2, apply_augment=True)
                    trainloader = torch.utils.data.DataLoader(randaug_trainset, batch_size=batch_size,
                                                          shuffle=True, num_workers=16)
                    
                    resnet = ResNet18().to(device)
                    accuracy, fixed_accuracy, _ = train_model(resnet, trainloader, testloader,
                                                         fixed_testloader, device)
                    
                    accuracies.append(accuracy)
                    fixed_accuracies.append(fixed_accuracy)
                    models.append(resnet)
                
                ensemble_preds = ensemble_predict(models, testloader, device)
                ensemble_fixed_preds = ensemble_predict(models, fixed_testloader, device)
                
                ensemble_acc = accuracy_score(testset.targets, ensemble_preds) * 100
                ensemble_fixed_acc = accuracy_score(
                    [testset.targets[i] for i in fixed_indices], ensemble_fixed_preds) * 100
                
                with open(os.path.join(method_dir, 'resnet', 'results.txt'), 'w') as f:
                    f.write("Individual Model Accuracies:\n")
                    for i, (acc, fixed_acc) in enumerate(zip(accuracies, fixed_accuracies)):
                        f.write(f"Model {i+1}:\n")
                        f.write(f"  Test Accuracy: {acc:.2f}%\n")
                        f.write(f"  Fixed Test Accuracy: {fixed_acc:.2f}%\n")
                    
                    f.write(f"\nMean Test Accuracy: {np.mean(accuracies):.2f}%\n")
                    f.write(f"Mean Fixed Test Accuracy: {np.mean(fixed_accuracies):.2f}%\n")
                    f.write(f"Ensemble Test Accuracy: {ensemble_acc:.2f}%\n")
                    f.write(f"Ensemble Fixed Test Accuracy: {ensemble_fixed_acc:.2f}%\n")
                
                print(f"ResNet - Mean Test Accuracy: {np.mean(accuracies):.2f}%")
                print(f"ResNet - Mean Fixed Test Accuracy: {np.mean(fixed_accuracies):.2f}%")
                print(f"ResNet - Ensemble Test Accuracy: {ensemble_acc:.2f}%")
                print(f"ResNet - Ensemble Fixed Test Accuracy: {ensemble_fixed_acc:.2f}%")

                loss_types = ['vanilla']
                if alpha is not None:
                    loss_types.extend(['balsoftmax', 'balsoftmax_entropy_5'])

                for loss_type in loss_types:
                    loss_dir = os.path.join(method_dir, 'cnn', loss_type)
                    os.makedirs(loss_dir, exist_ok=True)
                    
                    print(f"\nTraining CNN ensemble with {loss_type}...")
                    models = []
                    accuracies = []
                    fixed_accuracies = []
                    
                    for i, subset in enumerate(subsets):
                        print(f"\nTraining CNN {i+1}/{n}")
                        if loss_type in ['balsoftmax', 'balsoftmax_entropy_5']:
                            trainloader = torch.utils.data.DataLoader(subset, batch_size=batch_size,
                                                                  shuffle=True, num_workers=16)
                            if alpha is not None:
                                sample_per_class = np.bincount([trainset.targets[idx] for idx in subset.indices], minlength=10)
                        else:
                            randaug_trainset = augment_subset(subset, augmentation_type='randaug', n=2, apply_augment=True)
                            trainloader = torch.utils.data.DataLoader(randaug_trainset, batch_size=batch_size,
                                                                  shuffle=True, num_workers=16)
                            sample_per_class = None
                        
                        net = TinyCNN().to(device)
                        accuracy, fixed_accuracy, params = train_model(
                            net, trainloader, testloader, fixed_testloader, device,
                            os.path.join(loss_dir, f'model_params{i+1}.json'),
                            loss_type, sample_per_class
                        )
                        
                        accuracies.append(accuracy)
                        fixed_accuracies.append(fixed_accuracy)
                        models.append(net)
                    
                    ensemble_preds = ensemble_predict(models, testloader, device)
                    ensemble_fixed_preds = ensemble_predict(models, fixed_testloader, device)
                    
                    ensemble_acc = accuracy_score(testset.targets, ensemble_preds) * 100
                    ensemble_fixed_acc = accuracy_score(
                        [testset.targets[i] for i in fixed_indices], ensemble_fixed_preds) * 100
                    
                    with open(os.path.join(loss_dir, 'results.txt'), 'w') as f:
                        f.write("Individual Model Accuracies:\n")
                        for i, (acc, fixed_acc) in enumerate(zip(accuracies, fixed_accuracies)):
                            f.write(f"Model {i+1}:\n")
                            f.write(f"  Test Accuracy: {acc:.2f}%\n")
                            f.write(f"  Fixed Test Accuracy: {fixed_acc:.2f}%\n")
                        
                        f.write(f"\nMean Test Accuracy: {np.mean(accuracies):.2f}%\n")
                        f.write(f"Mean Fixed Test Accuracy: {np.mean(fixed_accuracies):.2f}%\n")
                        f.write(f"Ensemble Test Accuracy: {ensemble_acc:.2f}%\n")
                        f.write(f"Ensemble Fixed Test Accuracy: {ensemble_fixed_acc:.2f}%\n")
                    
                    print(f"CNN ({loss_type}) - Mean Test Accuracy: {np.mean(accuracies):.2f}%")
                    print(f"CNN ({loss_type}) - Mean Fixed Test Accuracy: {np.mean(fixed_accuracies):.2f}%")
                    print(f"CNN ({loss_type}) - Ensemble Test Accuracy: {ensemble_acc:.2f}%")
                    print(f"CNN ({loss_type}) - Ensemble Fixed Test Accuracy: {ensemble_fixed_acc:.2f}%")

if __name__ == '__main__':
    main()