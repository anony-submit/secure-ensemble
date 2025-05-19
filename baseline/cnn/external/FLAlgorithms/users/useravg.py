import torch
from FLAlgorithms.users.userbase import User

class UserAVG(User):
    def __init__(self,  args, id, model, train_data, test_data, use_adam=False):
        super().__init__(args, id, model, train_data, test_data, use_adam=use_adam)

    def update_label_counts(self, labels, counts):
        for label, count in zip(labels, counts):
            self.label_counts[int(label)] += count


    def clean_up_counts(self):
        del self.label_counts
        self.label_counts = {int(label):1 for label in range(self.unique_labels)}

    def train(self, glob_iter, personalized=False, lr_decay=True, count_labels=True):
        if hasattr(self, "train_data_logged") is False:
            label_list = []
            for x, y in self.trainloaderfull:
                label_list.extend(y.tolist())
            label_counts = {label: label_list.count(label) for label in sorted(set(label_list))}
            print(f"[Client {self.id}] Data label distribution: {label_counts}")
            self.train_data_logged = True
        self.clean_up_counts()
        self.model.train()
        for epoch in range(1, self.local_epochs + 1):
            self.model.train()
            for i in range(self.K):
                result = self.get_next_train_batch(count_labels=count_labels)
                X, y = result['X'], result['y']
                if count_labels:
                    self.update_label_counts(result['labels'], result['counts'])

                self.optimizer.zero_grad()
                X, y = X.to(self.device), y.to(self.device)
                if not getattr(self.model, "is_cnn_input", False):
                    X = X.view(X.size(0), -1)

                output = self.model(X)
                loss = self.loss(output, y)
                loss.backward()
                self.optimizer.step()

            self.clone_model_paramenter(self.model.parameters(), self.local_model)
            if personalized:
                self.clone_model_paramenter(self.model.parameters(), self.personalized_model_bar)
        if lr_decay:
            self.lr_scheduler.step(glob_iter)

