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
        """
        Initializes a Metric object.

        Args:
            name (str): The name of the metric.
        """
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
    Calculates the precision metric for a multi-class classification problem.

    Precision is defined as the ratio of true positive predictions to the total number of positive predictions made by the model.

    Args:
        None

    Returns:
        float: The precision score.
    """

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
    """
    Calculates the recall metric for a multi-class classification problem.

    Recall is defined as the ratio of true positive predictions to the total number of actual positive samples.

    Args:
        None

    Returns:
        float: The recall score.
    """

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
    """
    Calculates the F1 score metric for a multi-class classification problem.

    F1 score is defined as the harmonic mean of precision and recall.

    Args:
        None
    
    Returns:
        float: The F1 score.
    """
    
    def __init__(self):
        super().__init__("f1_score")

    def __call__(self, y_true, y_pred):
        precision_metric = Precision()
        recall_metric = Recall()
        precision = precision_metric(y_true, y_pred)
        recall = recall_metric(y_true, y_pred)
        return 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0