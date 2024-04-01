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
    """
    Precision metric calculates the ratio of true positive predictions to the total number of positive predictions.

    Args:
        None

    Returns:
        precision (float): The precision value.

    Example usage:
        precision = Precision()
        precision_value = precision(y_true, y_pred)
    """

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
    """
    Calculates the recall metric for binary classification tasks.

    Recall measures the ability of a model to find all the relevant cases (true positives) in a dataset.
    It is defined as the ratio of true positives to the sum of true positives and false negatives.

    Args:
        None

    Returns:
        float: The recall value.

    Example:
        >>> recall = Recall()
        >>> y_true = [1, 0, 1, 1, 0]
        >>> y_pred = [0.8, 0.2, 0.6, 0.9, 0.3]
        >>> recall_value = recall(y_true, y_pred)
        >>> print(recall_value)
        0.6666666666666666
    """

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
    """
    F1Score is a metric that calculates the F1 score, which is a measure of a model's accuracy.
    It combines precision and recall to provide a single metric that balances both measures.

    Args:
        None

    Returns:
        float: The F1 score value.

    Example:
        f1_score = F1Score()
        score = f1_score(y_true, y_pred)
    """

    def __init__(self):
        super().__init__("f1_score")

    def __call__(self, y_true, y_pred):
        precision_metric = Precision()
        recall_metric = Recall()
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        return f1_score
