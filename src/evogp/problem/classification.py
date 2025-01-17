import torch
from torch import Tensor
from . import BaseProblem
from evogp.tree import Forest

from sklearn.datasets import load_iris, load_wine, load_breast_cancer, load_digits


class Classification(BaseProblem):
    def __init__(self, dataset: str, multi_output=False):
        self.datapoints, self.labels = self.generate_data(dataset)
        self.onehot_labels = torch.zeros(
            self.labels.size(0), self.maximum + 1, device="cuda"
        )
        self.onehot_labels.scatter_(1, self.labels.long().unsqueeze(1), 1)
        self.multi_output = multi_output

    def generate_data(self, dataset: str):
        if dataset == "iris":
            X, y = load_iris(return_X_y=True)
        elif dataset == "wine":
            X, y = load_wine(return_X_y=True)
        elif dataset == "breast_cancer":
            X, y = load_breast_cancer(return_X_y=True)
        elif dataset == "digits":
            X, y = load_digits(return_X_y=True)
        else:
            raise ValueError("Invalid dataset")
        inputs = torch.tensor(X, dtype=torch.float32, device="cuda")
        labels = torch.tensor(y, dtype=torch.float32, device="cuda")
        self.maximum = int(torch.max(labels))
        return inputs, labels

    def transform(self, x: Tensor):
        x = torch.round(x + self.maximum / 2)
        return torch.clamp(x, 0, self.maximum).squeeze()

    def evaluate(self, forest: Forest):
        outputs = forest.batch_forward(self.datapoints)
        if not self.multi_output:
            y_pred = self.transform(outputs)
            return torch.sum(y_pred == self.labels, dim=1, dtype=torch.float32)
        else:
            eps = 1e-15
            class_prob = torch.clip(torch.softmax(outputs, dim=2), eps, 1 - eps)
            y_pred = torch.argmax(class_prob, dim=2)
            correct_num = torch.sum(y_pred == self.labels, dim=1, dtype=torch.float32)
            print(f"correct num: {int(correct_num.max())}")
            return (
                torch.sum(self.onehot_labels * torch.log(class_prob), dim=(1, 2))
                / self.onehot_labels.shape[0]
            )
