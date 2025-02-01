import torch
import torchvision
import torchvision.transforms as transforms
import numpy as np
import os
from multiprocessing import freeze_support
import multiprocessing as mp
from sklearn.metrics import accuracy_score
import pandas as pd

from models import NeuralNetwork, ResNet18
from data_utils import create_dirichlet_split
from train_utils import train_model, ensemble_predict

def get_sample_per_class(subset):
    labels = [label for _, label in subset]
    unique_labels, counts = np.unique(labels, return_counts=True)
    sample_per_class = np.zeros(10)
    sample_per_class[unique_labels] = counts
    return sample_per_class

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,)) 
    ])

    num_workers = min(mp.cpu_count() * 2, 16)
    print(f"Using {num_workers} workers")

    trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)
    testset = torchvision.datasets.FashionMNIST(root='./data', train=False,
                                       download=True, transform=transform)
    
    test_size = len(testset)
    fixed_test_indices = torch.randperm(test_size)[:100] 
    fixed_testset = torch.utils.data.Subset(testset, fixed_test_indices)
        
    test_data = []
    test_labels = []
    for idx in fixed_test_indices:  
        data, label = testset[idx]
        test_data.append(data.numpy().reshape(-1))
        test_labels.append(label)
    
    test_data = np.vstack(test_data)
    test_labels = np.array(test_labels)
    
    os.makedirs('test_data', exist_ok=True)
    df = pd.DataFrame(test_data)
    df['label'] = test_labels
    df.to_csv('test_data/fmnist_test.csv', index=False)

    n_values = [1, 2, 5, 10, 20]
    base_dir = 'results'
    os.makedirs(base_dir, exist_ok=True)

    testloader = torch.utils.data.DataLoader(testset, batch_size=128, 
                                           shuffle=False, num_workers=num_workers,
                                           pin_memory=True)
    fixed_testloader = torch.utils.data.DataLoader(fixed_testset, batch_size=128,
                                                 shuffle=False, num_workers=num_workers,
                                                 pin_memory=True)

    for n in n_values:
        print(f"\n=== Starting experiment with n={n} ===")
        n_dir = os.path.join(base_dir, f'n{n}')
        os.makedirs(n_dir, exist_ok=True)
        
        if n == 1:
            model_types = ['neuralnetwork', 'resnet']
            for model_type in model_types:
                print(f"\nTraining {model_type} model")
                
                if model_type == 'neuralnetwork':
                    net = NeuralNetwork().to(device)
                else:
                    net = ResNet18().to(device)
                
                trainloader = torch.utils.data.DataLoader(trainset, 
                    batch_size=128, shuffle=True, num_workers=num_workers,
                    pin_memory=True)
                
                if model_type == 'neuralnetwork':
                    model_save_path = os.path.join(n_dir, f'model_params_{model_type}.json')
                else:
                    model_save_path = None
                
                accuracy, fixed_accuracy, _ = train_model(net, trainloader, testloader, 
                                                     fixed_testloader, device, model_save_path)
                
                with open(os.path.join(n_dir, f'results_{model_type}.txt'), 'w') as f:
                    f.write(f"Full test accuracy: {accuracy:.2f}%\n")
                    f.write(f"Fixed test accuracy: {fixed_accuracy:.2f}%\n")
        else:
            split_methods = [
                ('balanced', None),
                ('dirichlet0.1', 0.1),
                ('dirichlet0.5', 0.5)
            ]
            
            for method_name, alpha in split_methods:
                print(f"\nStarting {method_name} split experiment")
                method_dir = os.path.join(n_dir, method_name)
                os.makedirs(method_dir, exist_ok=True)
                
                if alpha is None:
                    train_indices = torch.randperm(len(trainset))  
                    split_size = len(trainset) // n
                    subsets = [torch.utils.data.Subset(trainset, 
                        train_indices[i*split_size:(i+1)*split_size]) for i in range(n)]
                else:
                    heatmap_path = os.path.join(method_dir, 'distribution.png')
                    subsets, distribution = create_dirichlet_split(trainset, n, alpha, 
                                                               save_path=heatmap_path)
                    with open(os.path.join(method_dir, 'distribution.txt'), 'w') as f:
                        f.write('Class distribution per split:\n')
                        for i, dist in enumerate(distribution):
                            f.write(f"\nSplit {i+1}:\n")
                            for class_idx, count in enumerate(dist):
                                f.write(f"Class {class_idx}: {int(count)}\n")
                
                print("\nTraining ResNet models")
                resnet_dir = os.path.join(method_dir, 'resnet')
                os.makedirs(resnet_dir, exist_ok=True)
                resnet_models = []
                resnet_accuracies = []
                resnet_fixed_accuracies = []
                
                for i, subset in enumerate(subsets):
                    print(f"\nTraining ResNet model {i+1}/{n}")
                    trainloader = torch.utils.data.DataLoader(subset,
                        batch_size=128, shuffle=True, num_workers=num_workers,
                        pin_memory=True)
                    
                    net = ResNet18().to(device)
                    accuracy, fixed_accuracy, _ = train_model(net, trainloader, testloader,
                                                         fixed_testloader, device, None)
                    print(f"ResNet {i+1} accuracy (full test): {accuracy:.2f}%")
                    print(f"ResNet {i+1} accuracy (fixed test): {fixed_accuracy:.2f}%")
                    
                    resnet_accuracies.append(accuracy)
                    resnet_fixed_accuracies.append(fixed_accuracy)
                    resnet_models.append(net)

                with open(os.path.join(resnet_dir, 'results.txt'), 'w') as f:
                    f.write(f"Individual model accuracies (full test):\n")
                    for i, acc in enumerate(resnet_accuracies):
                        f.write(f"Model {i+1}: {acc:.2f}%\n")
                    f.write(f"\nAverage individual accuracy (full test): {np.mean(resnet_accuracies):.2f}%\n")
                    
                    f.write(f"\nIndividual model accuracies (fixed test):\n")
                    for i, acc in enumerate(resnet_fixed_accuracies):
                        f.write(f"Model {i+1}: {acc:.2f}%\n")
                    f.write(f"\nAverage individual accuracy (fixed test): {np.mean(resnet_fixed_accuracies):.2f}%\n")

                print("\nComputing ResNet ensemble predictions...")
                test_labels = np.array(testset.targets)
                fixed_test_labels = np.array([testset.targets[i.item()] for i in fixed_test_indices])
                
                soft_predictions = ensemble_predict(resnet_models, testloader, device)
                soft_predictions_fixed = ensemble_predict(resnet_models, fixed_testloader, device)
                
                soft_accuracy = accuracy_score(test_labels, soft_predictions) * 100
                soft_accuracy_fixed = accuracy_score(fixed_test_labels, soft_predictions_fixed) * 100
                
                print(f"Ensemble accuracy (soft voting, full test): {soft_accuracy:.2f}%")
                print(f"Ensemble accuracy (soft voting, fixed test): {soft_accuracy_fixed:.2f}%")
                
                with open(os.path.join(resnet_dir, 'ensemble_results.txt'), 'w') as f:
                    f.write(f"Ensemble accuracy (soft voting, full test): {soft_accuracy:.2f}%\n")
                    f.write(f"Ensemble accuracy (soft voting, fixed test): {soft_accuracy_fixed:.2f}%\n")
                
                print("\nTraining NeuralNetwork models")
                nn_dir = os.path.join(method_dir, 'neuralnetwork')
                os.makedirs(nn_dir, exist_ok=True)
                
                if alpha is None:
                    training_modes = ['vanilla']
                else:
                    training_modes = ['vanilla', 'balsoftmax', 'balsoftmax_entropy_small', 
                                    'balsoftmax_entropy_medium', 'balsoftmax_entropy_large']
                
                for mode in training_modes:
                    print(f"\nTraining NeuralNetwork models with {mode}")
                    mode_dir = os.path.join(nn_dir, mode)
                    os.makedirs(mode_dir, exist_ok=True)
                    
                    models = []
                    accuracies = []
                    fixed_accuracies = []
                    
                    for i, subset in enumerate(subsets):
                        print(f"\nTraining NeuralNetwork model {i+1}/{n}")
                        trainloader = torch.utils.data.DataLoader(subset,
                            batch_size=128, shuffle=True, num_workers=num_workers,
                            pin_memory=True)
                        
                        net = NeuralNetwork().to(device)
                        sample_per_class = get_sample_per_class(subset) if mode != 'vanilla' else None
                        
                        model_save_path = os.path.join(mode_dir, f'model_params{i+1}.json')
                        accuracy, fixed_accuracy, params = train_model(
                            net, trainloader, testloader, fixed_testloader, 
                            device, model_save_path, mode, sample_per_class)
                        
                        accuracies.append(accuracy)
                        fixed_accuracies.append(fixed_accuracy)

                        if params:
                            state_dict = {}
                            state_dict['conv.weight'] = torch.tensor(params['conv_params']['kernels']).view(4, 1, 2, 2)
                            state_dict['conv.bias'] = torch.tensor(params['conv_params']['bias'])
                            state_dict['fc1.weight'] = torch.tensor(params['fc1.weight'])
                            state_dict['fc1.bias'] = torch.tensor(params['fc1.bias'])
                            state_dict['fc2.weight'] = torch.tensor(params['fc2.weight'])
                            state_dict['fc2.bias'] = torch.tensor(params['fc2.bias'])
                            net.load_state_dict(state_dict)
                        models.append(net)
                    
                    with open(os.path.join(mode_dir, 'results.txt'), 'w') as f:
                        f.write(f"Individual model accuracies (full test):\n")
                        for i, acc in enumerate(accuracies):
                            f.write(f"Model {i+1}: {acc:.2f}%\n")
                        f.write(f"\nAverage individual accuracy (full test): {np.mean(accuracies):.2f}%\n")
                        
                        f.write(f"\nIndividual model accuracies (fixed test):\n")
                        for i, acc in enumerate(fixed_accuracies):
                            f.write(f"Model {i+1}: {acc:.2f}%\n")
                        f.write(f"\nAverage individual accuracy (fixed test): {np.mean(fixed_accuracies):.2f}%\n")
                    
                    print("\nComputing ensemble predictions...")
                    test_labels = np.array(testset.targets)
                    fixed_test_labels = np.array([testset.targets[i.item()] for i in fixed_test_indices])
                    
                    soft_predictions = ensemble_predict(models, testloader, device)
                    soft_predictions_fixed = ensemble_predict(models, fixed_testloader, device)
                    
                    soft_accuracy = accuracy_score(test_labels, soft_predictions) * 100
                    soft_accuracy_fixed = accuracy_score(fixed_test_labels, soft_predictions_fixed) * 100
                    
                    print(f"Ensemble accuracy (soft voting, full test): {soft_accuracy:.2f}%")
                    print(f"Ensemble accuracy (soft voting, fixed test): {soft_accuracy_fixed:.2f}%")
                    
                    with open(os.path.join(mode_dir, 'ensemble_results.txt'), 'w') as f:
                        f.write(f"Ensemble accuracy (soft voting, full test): {soft_accuracy:.2f}%\n")
                        f.write(f"Ensemble accuracy (soft voting, fixed test): {soft_accuracy_fixed:.2f}%\n")
                        
if __name__ == '__main__':
    freeze_support()
    main()