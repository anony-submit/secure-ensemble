import sys
import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../external/FedGen")))

from models_mnist_nn1 import MnistNN1
from data_split_loader import create_custom_data_loaders, create_balanced_data_loaders
from FLAlgorithms.servers.serveravg import FedAvg
from FLAlgorithms.servers.serverFedProx import FedProx

def get_preferred_device():
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        print("[INFO] ✅ Using device: MPS (Apple Metal GPU backend)")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
        print(f"[INFO] ✅ Using device: CUDA ({torch.cuda.get_device_name(device)})")
    else:
        device = torch.device("cpu")
        print("[INFO] ⚠️ Using device: CPU (No MPS or CUDA available)")
    return device

class DummyArgs:
    def __init__(self, algorithm, num_users, alpha):
        self.dataset = f'mnist_alpha{alpha}_n{num_users}'
        self.model = 'mnist_nn1'
        self.algorithm = algorithm
        self.batch_size = 32
        self.local_epochs = 32
        self.num_users = num_users
        self.learning_rate = 0.001
        self.num_glob_iters = 300
        self.device = get_preferred_device()
        self.embedding = 0
        self.result_path = './results'
        self.train = 3
        self.K = 5
        self.personal_learning_rate = 0.001
        self.times = 1
        self.beta = 1.0
        self.lamda = 1
        self.gen_batch_size = self.batch_size

def create_server(args, model, user_datasets):
    if args.algorithm == 'FedAvg':
        from FLAlgorithms.users.useravg import UserAVG
        return FedAvg(args, model, seed=0, user_datasets=user_datasets)
    elif args.algorithm == 'FedProx':
        from FLAlgorithms.users.userFedProx import UserFedProx
        return FedProx(args, model, seed=0, user_datasets=user_datasets)
    else:
        raise NotImplementedError

def load_mnist_test_from_csv(path="mnist_test.csv", batch_size=32):
    df = pd.read_csv(path, header=None)
    labels = torch.tensor(df.iloc[:, -1].values).long()
    images = torch.tensor(df.iloc[:, :-1].values).float().reshape(-1, 1, 28, 28)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)

def evaluate_on_central_test(model, test_loader, device='cpu'):
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            x = x.view(x.size(0), -1)
            outputs = model(x)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    accuracy = correct / total
    print(f"[Central Evaluation] Correct: {correct} / {total} | Accuracy: {accuracy:.4f}")
    return correct, total, accuracy

def run_all_experiments():
    results_path = "results_accuracy.txt"
    os.makedirs(os.path.dirname(results_path) or ".", exist_ok=True)
    print(f"[*] Results will be saved to: {results_path}")
    with open(results_path, "w") as f:
        f.write(f"{'Algorithm':<10}\t{'NumUsers':<9}\t{'Alpha':<12}\t{'BestRound':<10}\t{'Correct/Total':<15}\t{'Accuracy':<10}\n")

        # for algorithm in ["FedAvg", "FedProx"]:
        for algorithm in ["FedProx"]:
            for num_users in [2, 5, 10]:
                for alpha in ["balanced", "0.5", "0.1"]:
                    print(f"\n[=] Starting experiment: {algorithm} | Users: {num_users} | Alpha: {alpha}")
                    args = DummyArgs(algorithm, num_users, alpha)

                    if alpha == "balanced":
                        print("[*] Loading balanced user data...")
                        user_loaders = create_balanced_data_loaders(num_users, batch_size=args.batch_size)
                    else:
                        dist_path = f"./data_split/n{num_users}_dirichlet{alpha}.txt"
                        print(f"[*] Loading Dirichlet user data from {dist_path}...")
                        user_loaders = create_custom_data_loaders(dist_path, num_users, batch_size=args.batch_size)

                    user_datasets = [list(loader.dataset) for loader in user_loaders]
                    model = (MnistNN1(), args.model)
                    server = create_server(args, model, user_datasets)

                    test_loader = load_mnist_test_from_csv(batch_size=args.batch_size)

                    print("[*] Starting federated training...")
                    best_accuracy, best_round = server.train(args, test_loader=test_loader)
                    print("[*] Training completed.")

                    print("[*] Evaluating global model on central test set...")
                    correct, total, accuracy = evaluate_on_central_test(server.model, test_loader, device=args.device)

                    f.write(f"{algorithm:<10}\t{num_users:<9}\t{alpha:<12}\t{best_round:<10}\t{correct}/{total:<15}\t{accuracy:.4f}\n")
                    f.flush()
                    print("[*] Evaluation result logged.\n")


if __name__ == "__main__":
    run_all_experiments()
