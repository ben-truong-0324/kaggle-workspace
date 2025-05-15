from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

def get_sklearn_model(name: str, **kwargs):
    if name == "decision_tree":
        return DecisionTreeClassifier(**kwargs)
    elif name == "random_forest":
        return RandomForestClassifier(**kwargs)
    elif name == "logistic_regression":
        return LogisticRegression(**kwargs)
    else:
        raise ValueError(f"Unknown sklearn model: {name}")


def get_nn_model(input_dim: int, **kwargs):
    import torch.nn as nn

    class SimpleNN(nn.Module):
        def __init__(self, input_dim, hidden=64):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden),
                nn.ReLU(),
                nn.Linear(hidden, 1)
            )

        def forward(self, x):
            return self.net(x)

    return SimpleNN(input_dim, **kwargs)
