import torch
from abc import ABC, abstractmethod

class Metric(ABC):
    """
    Abstract base class for defining metrics.

    Attributes:
        name (str): The name of the metric.

    Methods:
        __call__(y_true, y_pred): Abstract method to calculate the metric.
        __str__(): Returns the name of the metric.
    """

    def __init__(self, name):
        self.name = name

    @abstractmethod
    def __call__(self, y_true, y_pred):
        """
        Abstract method to calculate the metric.

        Args:
            y_true: The true values.
            y_pred: The predicted values.

        Returns:
            The calculated metric value.
        """
        raise NotImplementedError

    def __str__(self):
        """
        Returns the name of the metric.

        Returns:
            The name of the metric.
        """
        return self.name


class Accuracy(Metric):
    """
    Calculates the accuracy metric for classification tasks.

    The accuracy is defined as the ratio of correctly predicted samples to the total number of samples.

    Args:
        None

    Returns:
        float: The accuracy value.

    Example:
        >>> accuracy = Accuracy()
        >>> y_true = torch.tensor([0, 1, 1, 0])
        >>> y_pred = torch.tensor([[0.9, 0.1], [0.2, 0.8], [0.3, 0.7], [0.6, 0.4]])
        >>> acc = accuracy(y_true, y_pred)
        >>> print(acc)
        0.75
    """

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
        with torch.no_grad():
            _, predictions = torch.max(y_pred, 1)
            unique_classes = torch.unique(y_true)
            precision_sum = 0
            for cls in unique_classes:
                true_positive = (predictions == cls)[y_true == cls].sum().item()
                predicted_positive = (predictions == cls).sum().item()
                precision_sum += true_positive / predicted_positive if predicted_positive > 0 else 0
            return precision_sum / len(unique_classes)

class Recall(Metric):
    def __init__(self):
        super().__init__("recall")

    def __call__(self, y_true, y_pred):
        with torch.no_grad():
            _, predictions = torch.max(y_pred, 1)
            unique_classes = torch.unique(y_true)
            recall_sum = 0
            for cls in unique_classes:
                true_positive = (predictions == cls)[y_true == cls].sum().item()
                actual_positive = (y_true == cls).sum().item()
                recall_sum += true_positive / actual_positive if actual_positive > 0 else 0
            return recall_sum / len(unique_classes)

class F1Score(Metric):
    def __init__(self):
        super().__init__("f1_score")

    def __call__(self, y_true, y_pred):
        precision_metric = Precision()
        recall_metric = Recall()
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0