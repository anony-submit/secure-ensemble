import os
import numpy as np
import torch
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from logistic_model import LogisticRegressionModel

torch.cuda.is_available = lambda: False

class Args:
    def __init__(self):
        self.epochs = 32
        self.rounds = 100
        self.lr = 0.001
        self.mu = 0.1
        self.algorithm = None

skip_config = {
    "wdbc": { "vertical": [False, False, False, False] },
    "heart_disease": { "vertical": [False, False, False, True] },
    "pima": { "vertical": [False, False, True, True] },
}

def load_dataset(name):
    if name == "wdbc":
        data = pd.read_csv("data/wdbc/wdbc.data", header=None)
        X = data.iloc[:, 2:]
        y = (data.iloc[:, 1] == 'M').astype(int)
    elif name == "heart_disease":
        data = pd.read_csv("data/heart_disease/Heart_disease_cleveland.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    else:
        data = pd.read_csv("data/pima/diabetes.csv")
        X = data.iloc[:, :-1]
        y = data.iloc[:, -1]
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, y.to_numpy()

def split_data(X, y, n_clients, method, seed=42):
    np.random.seed(seed)
    if method == "balanced":
        indices = np.random.permutation(len(y))
        return [(X[i], y[i]) for i in np.array_split(indices, n_clients)]
    elif method.startswith("dirichlet"):
        alpha = float(method.split("_")[1])
        class_indices = {label: np.where(y == label)[0] for label in np.unique(y)}
        client_indices = [[] for _ in range(n_clients)]
        for label, indices in class_indices.items():
            proportions = np.random.dirichlet([alpha] * n_clients)
            split = np.split(np.random.permutation(indices), (proportions.cumsum()[:-1] * len(indices)).astype(int))
            for cid, part in enumerate(split):
                client_indices[cid].extend(part)
        return [(X[inds], y[inds]) for inds in client_indices]
    else:
        raise ValueError("Unknown method")

def split_dataset_vertically(X, n_splits, method='balanced', alpha=None, random_state=42):
    if method == 'balanced':
        return split_dataset_vertically_balanced(X, n_splits, random_state)
    elif method in ['dirichlet_0.5', 'dirichlet_0.1']:
        alpha = 0.5 if method == 'dirichlet_0.5' else 0.1
        return split_dataset_vertically_dirichlet(X, n_splits, alpha, random_state)
    else:
        raise ValueError(f"Unknown vertical method: {method}")

def split_dataset_vertically_balanced(X, n_splits, random_state=42):
    np.random.seed(random_state)
    feature_count = X.shape[1]
    base_size = feature_count // n_splits
    remainder = feature_count % n_splits
    split_sizes = [base_size + 1 if i < remainder else base_size for i in range(n_splits)]
    feature_indices = np.random.permutation(feature_count)
    X_splits, index_splits = [], []
    start_idx = 0
    for size in split_sizes:
        end_idx = start_idx + size
        selected_features = feature_indices[start_idx:end_idx]
        X_splits.append(X[:, selected_features])
        index_splits.append(selected_features)
        start_idx = end_idx
    return X_splits, index_splits

def split_dataset_vertically_dirichlet(X, n_splits, alpha, random_state=42):
    np.random.seed(random_state)
    feature_count = X.shape[1]
    proportions = np.random.dirichlet([alpha] * n_splits)
    split_sizes = np.maximum((proportions * feature_count).astype(int), 1)
    while split_sizes.sum() > feature_count:
        largest = np.argmax(split_sizes)
        split_sizes[largest] -= 1
    split_sizes[-1] += feature_count - split_sizes.sum()
    feature_indices = np.random.permutation(feature_count)
    X_splits, index_splits = [], []
    start = 0
    for size in split_sizes:
        end = start + size
        selected = feature_indices[start:end]
        X_splits.append(X[:, selected])
        index_splits.append(selected)
        start = end
    return X_splits, index_splits

def prepare_loader(X, y):
    if len(X) == 0 or len(y) == 0:
        return None
    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y))
    return DataLoader(dataset, batch_size=32, shuffle=True)

def train_client(model, loader, args, global_params=None):
    if loader is None:
        return None
    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    loss_fn = torch.nn.BCELoss()
    for _ in range(args.epochs):
        for xb, yb in loader:
            xb, yb = xb, yb.unsqueeze(1)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if args.algorithm == "fedprox" and global_params is not None:
                prox_term = 0.0
                for name, param in model.named_parameters():
                    if "bias" in name:
                        continue
                    prox_term += ((param - global_params[name]) ** 2).sum()
                loss += (args.mu / 2) * prox_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

def get_full_weights(model, feature_indices, total_dim):
    W = torch.zeros(total_dim, 1)
    local_W = list(model.parameters())[0].detach().view(-1)
    b = list(model.parameters())[1].detach()
    for i, idx in enumerate(feature_indices):
        W[idx, 0] = local_W[i]
    return W, b

def load_global_weights(model, global_W, global_b, feat_idxs):
    with torch.no_grad():
        model.linear.bias.copy_(global_b)
        for i, idx in enumerate(feat_idxs):
            model.linear.weight[0, i] = global_W[idx, 0]

def run_experiment(dataset_name, split_type, method, n_clients, algo, output_file):
    if split_type == "vertical":
        index_map = {n: i for i, n in enumerate([2, 5, 10, 20])}
        if dataset_name in skip_config and n_clients in index_map:
            if skip_config[dataset_name][split_type][index_map[n_clients]]:
                return

    total_acc = 0.0
    trials = 10
    for seed in range(trials):
        X, y = load_dataset(dataset_name)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed)
        args = Args()
        args.algorithm = algo
        total_dim = X.shape[1]

        if split_type == "horizontal":
            splits = split_data(X_train, y_train, n_clients, method, seed=seed)
            splits = [(X_c, y_c, np.arange(total_dim)) for X_c, y_c in splits]
        else:
            X_splits, feature_indices = split_dataset_vertically(X_train, n_clients, method, random_state=seed)
            splits = [(X_splits[i], y_train, feature_indices[i]) for i in range(n_clients)]

        global_W = torch.zeros(total_dim, 1)
        global_b = torch.zeros(1)

        for _ in range(args.rounds):
            local_ws, local_bs = [], []
            sample_counts = []
            for X_c, y_c, feat_idxs in splits:
                input_dim = X_c.shape[1]
                model = LogisticRegressionModel(input_dim)
                load_global_weights(model, global_W, global_b, feat_idxs)

                global_state_dict = model.state_dict()
                global_params = {k: v.clone().detach() for k, v in global_state_dict.items()}

                loader = prepare_loader(X_c, y_c)
                train_client(model, loader, args, global_params if algo == "fedprox" else None)

                w_full, b = get_full_weights(model, feat_idxs, total_dim)
                local_ws.append(w_full)
                local_bs.append(b)
                sample_counts.append(len(y_c))

            total_samples = sum(sample_counts)
            global_W = sum(w * (n / total_samples) for w, n in zip(local_ws, sample_counts))
            global_b = sum(b * (n / total_samples) for b, n in zip(local_bs, sample_counts))

        with torch.no_grad():
            X_test_tensor = torch.FloatTensor(X_test)
            logits = torch.sigmoid(X_test_tensor @ global_W + global_b)
            preds = logits.numpy().squeeze() > 0.5
            acc = accuracy_score(y_test, preds)
            total_acc += acc

    avg_acc = total_acc / trials
    with open(output_file, "a") as f:
        f.write(f"{dataset_name},{split_type},{method},n={n_clients},{algo},acc={avg_acc:.4f}\n")
    print(f"{dataset_name},{split_type},{method},n={n_clients},{algo},acc={avg_acc:.4f}")

if __name__ == "__main__":
    datasets = ["wdbc", "heart_disease", "pima"]
    split_types = ["horizontal", "vertical"]
    methods = ["balanced", "dirichlet_0.5", "dirichlet_0.1"]
    n_values = [2, 5, 10, 20]
    algos = ["fedavg", "fedprox"] 
    for algo in algos:
        output_file = f"logistic_results_{algo}.txt"
        for dataset in datasets:
            for split_type in split_types:
                for method in methods:
                    for n in n_values:
                        run_experiment(dataset, split_type, method, n, algo, output_file)
