import torch
from abc import ABC, abstractmethod

class Metric(ABC):
    """Abstract class for metrics"""

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, y_true, y_pred):
        raise NotImplementedError

    def __str__(self):
        return self.name


class Accuracy(Metric):
    def __init__(self):
        super().__init__("accuracy")

    def __call__(self, y_true, y_pred):
        correct = 0
        total = 0
        with torch.no_grad():
            _, predicted = torch.max(y_pred, 1)
            total += y_true.size(0)
            correct += (predicted == y_true).sum().item()
        return correct / total


class Precision(Metric):
    def __init__(self):
        super().__init__("precision")

    def __call__(self, y_true, y_pred):
        _, predictions = torch.max(y_pred, 1)
        # predictions = y_pred > threshold  # For binary classification
        true_positive = (predictions == y_true)[predictions == 1].sum().item()
        predicted_positive = (predictions == 1).sum().item()
        precision = true_positive / predicted_positive if predicted_positive > 0 else 0
        return precision


class Recall(Metric):
    def __init__(self):
        super().__init__("recall")

    def __call__(self, y_true, y_pred):
        _, predictions = torch.max(y_pred, 1)
        # predictions = y_pred > threshold  # For binary classification
        true_positive = (predictions == y_true)[y_true == 1].sum().item()
        actual_positive = (y_true == 1).sum().item()
        recall = true_positive / actual_positive if actual_positive > 0 else 0
        return recall


class F1Score(Metric):
    def __init__(self):
        super().__init__("f1_score")

    def __call__(self, y_true, y_pred):
        precision_metric = Precision()
        recall_metric = Recall()
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
