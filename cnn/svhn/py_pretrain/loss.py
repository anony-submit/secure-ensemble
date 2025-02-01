import torch
import torch.nn as nn
import torch.nn.functional as F

class BalancedSoftmaxLoss(nn.Module):
    def __init__(self, num_classes, sample_per_class):
        super().__init__()
        sample_per_class = torch.FloatTensor(sample_per_class)
        sample_per_class[sample_per_class == 0] = 1
        self.sample_per_class = sample_per_class
        self.num_classes = num_classes
        
    def forward(self, input, target):
        with torch.no_grad():
            one_hot = torch.zeros_like(input)
            one_hot[range(len(target)), target] = 1
            sample_per_class = self.sample_per_class.to(input.device)
            
        # numerator = torch.exp(input) * sample_per_class.view(1, -1)
        # denominator = numerator.sum(dim=1, keepdim=True)
        
        # loss = -torch.sum(torch.log(numerator / (denominator + 1e-7)) * one_hot) / len(target)
        
        logits = input + torch.log(sample_per_class.view(1, -1) + 1e-7)
        log_sum_exp = torch.logsumexp(logits, dim=1, keepdim=True)
        loss = -torch.sum((logits - log_sum_exp) * one_hot) / len(target)
        
        return loss

class BalancedSoftmaxLossWithEntropyReg(nn.Module):
    def __init__(self, num_classes, sample_per_class, entropy_weight=0.1):
        super().__init__()
        self.balanced_loss = BalancedSoftmaxLoss(num_classes, sample_per_class)
        self.entropy_weight = entropy_weight
        
    def forward(self, input, target):
        balanced_loss = self.balanced_loss(input, target)
        probs = F.softmax(input, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-7), dim=1).mean()
        total_loss = balanced_loss - self.entropy_weight * entropy
        return total_loss