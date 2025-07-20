import torch
from torchmetrics import Accuracy, Precision, Recall, F1Score

class Metrics:
    def __init__(self, task='binary', num_classes=2, average='macro', device='cpu'):
        self.task = task
        if task == 'binary':
            self.metrics = {
                'accuracy': Accuracy(task='binary').to(device),
                'precision': Precision(task='binary').to(device),
                'recall': Recall(task='binary').to(device),
                'f1': F1Score(task='binary').to(device)
            }
        elif task == 'multiclass':
            self.metrics = {
                'accuracy': Accuracy(task='multiclass', num_classes=num_classes, average=average).to(device),
                'precision': Precision(task='multiclass', num_classes=num_classes, average=average).to(device),
                'recall': Recall(task='multiclass', num_classes=num_classes, average=average).to(device),
                'f1': F1Score(task='multiclass', num_classes=num_classes, average=average).to(device)
            }
        else:
            raise ValueError("task must be 'binary' or 'multiclass'")

    def update(self, preds, targets):
        for metric in self.metrics.values():
            metric.update(preds, targets)

    def compute(self):
        return {name: metric.compute().item() for name, metric in self.metrics.items()}

    def reset(self):
        for metric in self.metrics.values():
            metric.reset()

    def print(self):
        results = self.compute()
        print(f"Metrics ({self.task} classification):")
        for name, value in results.items():
            print(f"  {name.capitalize()}: {value:.4f}")

# Example usage:
# metrics = Metrics(task='binary')
# metrics.update(preds, targets)
# metrics.print()
#
# metrics = Metrics(task='multiclass', num_classes=3)
# metrics.update(preds, targets)
# metrics.print()