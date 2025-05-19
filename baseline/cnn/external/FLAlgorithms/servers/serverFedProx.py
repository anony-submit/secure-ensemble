from FLAlgorithms.users.userFedProx import UserFedProx
from FLAlgorithms.servers.serverbase import Server
from utils.model_utils import read_data, read_user_data
import torch
import copy


class FedProx(Server):
    def __init__(self, args, model, seed, user_datasets=None):
        super().__init__(args, model, seed)

        if user_datasets is not None:
            print("Using externally provided user datasets.")
            self.users = []
            for i, data in enumerate(user_datasets):
                user = UserFedProx(args, i, model, data, data, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples
        else:
            data = read_data(args.dataset)
            total_users = len(data[0])
            print("Users in total: {}".format(total_users))
            for i in range(total_users):
                id, train_data, test_data = read_user_data(i, data, dataset=args.dataset)
                user = UserFedProx(args, id, model, train_data, test_data, use_adam=False)
                self.users.append(user)
                self.total_train_samples += user.train_samples

        print("Number of users / total users:", args.num_users, " / ", len(self.users))
        print("Finished creating FedProx server.")

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

            for user in self.selected_users:
                if hasattr(user, 'train'):
                    user.train(glob_iter, personalized=getattr(self, 'personalized', False))
                else:
                    user.train(glob_iter)

            if hasattr(self, 'aggregate_parameters'):
                self.aggregate_parameters()

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

    # def train(self, args):
    #     for glob_iter in range(self.num_glob_iters):
    #         # print("\n\n-------------Round number: ", glob_iter, " -------------\n\n")
    #         self.selected_users = self.select_users(glob_iter, self.num_users)
    #         self.send_parameters()
    #         self.evaluate()
    #         for user in self.selected_users:
    #             user.train(glob_iter)
    #         self.aggregate_parameters()
    #     self.save_results(args)
    #     self.save_model()