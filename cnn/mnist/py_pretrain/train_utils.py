import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import json
import numpy as np
from models import ResNet18
from loss import BalancedSoftmaxLoss, BalancedSoftmaxLossWithEntropyReg

def train_model(net, trainloader, testloader, fixed_testloader, device, model_save_path=None, 
                training_mode='vanilla', sample_per_class=None):
    if training_mode == 'vanilla':
        criterion = nn.CrossEntropyLoss()
    elif training_mode == 'balsoftmax':
        criterion = BalancedSoftmaxLoss(10, sample_per_class)
    elif training_mode == 'balsoftmax_entropy_small':
        criterion = BalancedSoftmaxLossWithEntropyReg(10, sample_per_class, entropy_weight=0.05)
    elif training_mode == 'balsoftmax_entropy_medium':
        criterion = BalancedSoftmaxLossWithEntropyReg(10, sample_per_class, entropy_weight=0.1)
    elif training_mode == 'balsoftmax_entropy_large':
        criterion = BalancedSoftmaxLossWithEntropyReg(10, sample_per_class, entropy_weight=0.2)
    else:
        raise ValueError(f"Unknown training mode: {training_mode}")
        
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)
    
    if isinstance(net, ResNet18):
        epochs = 10
    else:
        epochs = 30
    
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)
    best_acc = 0
    best_params = None
    best_fixed_acc = 0

    epoch_pbar = tqdm(range(epochs), desc='Training Epochs')
    for epoch in epoch_pbar:
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        batch_pbar = tqdm(trainloader, leave=False, desc=f'Epoch {epoch + 1}')
        for data in batch_pbar:
            inputs, labels = data[0].to(device), data[1].to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
            
            batch_pbar.set_postfix({'Loss': f'{loss.item():.3f}'})
        
        train_accuracy = 100 * correct_train / total_train
        
        net.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        correct_fixed = 0
        total_fixed = 0
        
        with torch.no_grad():
            for data in testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                loss = nn.CrossEntropyLoss()(outputs, labels)
                test_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_test += labels.size(0)
                correct_test += (predicted == labels).sum().item()
            
            for data in fixed_testloader:
                images, labels = data[0].to(device), data[1].to(device)
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total_fixed += labels.size(0)
                correct_fixed += (predicted == labels).sum().item()
        
        test_accuracy = 100 * correct_test / total_test
        fixed_accuracy = 100 * correct_fixed / total_fixed
        
        epoch_pbar.set_postfix({
            'Train Loss': f'{running_loss/len(trainloader):.3f}',
            'Train Acc': f'{train_accuracy:.2f}%',
            'Test Acc': f'{test_accuracy:.2f}%',
            'Fixed Test Acc': f'{fixed_accuracy:.2f}%'
        })
        
        if test_accuracy > best_acc:
            best_acc = test_accuracy
            best_fixed_acc = fixed_accuracy
            if not isinstance(net, ResNet18) and model_save_path:
                best_params = {name: param.cpu().detach().numpy().tolist() 
                              for name, param in net.state_dict().items()}
            elif isinstance(net, ResNet18):
                best_params = None
        
        scheduler.step()
    
    if model_save_path and best_params and not isinstance(net, ResNet18):
        with open(model_save_path, 'w') as f:
            json.dump(best_params, f)
    
    return best_acc, best_fixed_acc, best_params

def ensemble_predict(models, testloader, device):
    all_predictions = []
    
    with torch.no_grad():
        for data in testloader:
            images = data[0].to(device)
            outputs = [model(images) for model in models]
            avg_outputs = sum(outputs) / len(outputs)
            _, predicted = torch.max(avg_outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
    
    return np.array(all_predictions)