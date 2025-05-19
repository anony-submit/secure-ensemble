from FLAlgorithms.users.useravg import UserAVG
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import numpy as np
import time
import torch
import copy

class FedAvg(Server):
    def __init__(self, args, model, seed, user_datasets=None):
        super().__init__(args, model, seed)

        if user_datasets is not None:
            print("Using externally provided user datasets.")
            self.users = []
            for i, data in enumerate(user_datasets):
                user = UserAVG(args, i, model, data, data, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        else:
            data = read_data(args.dataset)
            total_users = len(data[0])
            self.use_adam = 'adam' in self.algorithm.lower()
            print("Users in total: {}".format(total_users))

            for i in range(total_users):
                id, train_data, test_data = read_user_data(i, data, dataset=args.dataset)
                user = UserAVG(args, id, model, train_data, test_data, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples

        print("Number of users / total users:", args.num_users, " / ", len(self.users))
        print("Finished creating FedAvg server.")

    # def train(self, args):
    #     for glob_iter in range(self.num_glob_iters):
    #         self.selected_users = self.select_users(glob_iter,self.num_users)
    #         self.send_parameters(mode=self.mode)
    #         self.evaluate()
    #         self.timestamp = time.time()
    #         for user in self.selected_users:
    #             user.train(glob_iter, personalized=self.personalized)
    #         curr_timestamp = time.time()
    #         train_time = (curr_timestamp - self.timestamp) / len(self.selected_users)
    #         self.metrics['user_train_time'].append(train_time)
    #         if self.personalized:
    #             print("Evaluate personal model\n")
    #             self.evaluate_personalized_model()
    #         self.timestamp = time.time()
    #         self.aggregate_parameters()
    #         curr_timestamp = time.time()
    #         agg_time = curr_timestamp - self.timestamp
    #         self.metrics['server_agg_time'].append(agg_time)
    #     self.save_results(args)
    #     self.save_model()

    @staticmethod
    def evaluate_on_central_test(model, test_loader, device):
        model.to(device)
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in test_loader:
                x, y = x.to(device), y.to(device)
                if hasattr(model, "is_cnn_input") and not model.is_cnn_input:
                    x = x.view(x.size(0), -1)
                outputs = model(x)
                preds = torch.argmax(outputs, dim=1)
                correct += (preds == y).sum().item()
                total += y.size(0)
        accuracy = correct / total
        print(f"[Central Evaluation] Correct: {correct} / {total} | Accuracy: {accuracy:.4f}")
        return correct, total, accuracy

    def train(self, args, test_loader=None):
        best_accuracy = 0.0
        best_round = -1
        best_model_state = None

        for glob_iter in range(self.num_glob_iters):
            self.selected_users = self.select_users(glob_iter, args.K)
            if hasattr(self, 'send_parameters'):
                self.send_parameters(mode=self.mode if hasattr(self, 'mode') else None)

            self.timestamp = time.time()
            for user in self.selected_users:
                if hasattr(user, 'train'):
                    user.train(glob_iter, personalized=getattr(self, 'personalized', False))
                else:
                    user.train(glob_iter)
            train_time = (time.time() - self.timestamp) / len(self.selected_users)
            self.metrics['user_train_time'].append(train_time)

            if self.personalized:
                self.evaluate_personalized_model()

            self.timestamp = time.time()
            if hasattr(self, 'aggregate_parameters'):
                self.aggregate_parameters()
            agg_time = time.time() - self.timestamp
            self.metrics['server_agg_time'].append(agg_time)

            if test_loader is not None:
                correct, total, accuracy = self.evaluate_on_central_test(self.model, test_loader, device=args.device)
                print(f"[Round {glob_iter}] Accuracy: {accuracy:.4f} ({correct}/{total})")
                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_round = glob_iter
                    best_model_state = copy.deepcopy(self.model.state_dict())

        if test_loader is not None:
            print(f"[*] Best Accuracy: {best_accuracy:.4f} at Round {best_round}")
            if best_model_state:
                self.model.load_state_dict(best_model_state)

        self.save_results(args)
        self.save_model()

        if test_loader is not None:
            return best_accuracy, best_round
        else:
            return None, None
