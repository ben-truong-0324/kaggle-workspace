# scikit-learn classifiers
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# gradient boosting frameworks
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

def get_sklearn_model(name: str, **kwargs):
    if name == "decision_tree":
        return DecisionTreeClassifier(**kwargs)
    elif name == "random_forest":
        return RandomForestClassifier(**kwargs)
    elif name == "logistic_regression":
        return LogisticRegression(**kwargs)
    elif name == "svm":
        return SVC(**kwargs)
    elif name == "naive_bayes":
        return GaussianNB(**kwargs)
    elif name == "knn":
        return KNeighborsClassifier(**kwargs)
    elif name == "xgboost":
        return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **kwargs)
    elif name == "lightgbm":
        return LGBMClassifier(**kwargs)
    else:
        raise ValueError(f"Unknown sklearn model: {name}")

def get_nn_model(input_dim: int, output_dim: int = 1, **kwargs):
    import torch.nn as nn

    hidden = kwargs.get("hidden", 64)
    dropout = kwargs.get("dropout", 0.0)
    activation_name = kwargs.get("activation", "relu")
    batch_norm = kwargs.get("batch_norm", False)
    lr = kwargs.get("lr", 0.01)
    model_type = kwargs.get("model_type", "mlp")
    num_layers = kwargs.get("num_layers", 2)  # Default to 2 layers

    activations = {
        "relu": nn.ReLU(),
        "tanh": nn.Tanh(),
        "leaky_relu": nn.LeakyReLU(),
        "sigmoid": nn.Sigmoid()
    }
    activation_fn = activations.get(activation_name, nn.ReLU())

    # === MLP ===
    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []

            # First layer
            layers.append(nn.Linear(input_dim, hidden))
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            layers.append(activation_fn)
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            # Hidden layers
            for _ in range(num_layers - 1):
                layers.append(nn.Linear(hidden, hidden))
                if batch_norm:
                    layers.append(nn.BatchNorm1d(hidden))
                layers.append(activation_fn)
                if dropout > 0:
                    layers.append(nn.Dropout(dropout))

            # Output layer
            layers.append(nn.Linear(hidden, output_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    # === CNN and LSTM definitions unchanged ===
    class CNN1D(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv1d(16, 32, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            )
            self.fc = nn.Sequential(
                nn.Linear(32, output_dim)
            )

        def forward(self, x):
            x = x.unsqueeze(1)
            x = self.conv(x).squeeze(-1)
            return self.fc(x)

    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, batch_first=True)
            self.fc = nn.Linear(hidden, output_dim)

        def forward(self, x):
            x = x.unsqueeze(1)
            _, (hn, _) = self.lstm(x)
            return self.fc(hn[-1])

    if model_type == "mlp":
        model = MLP()
    elif model_type == "cnn":
        model = CNN1D()
    elif model_type == "lstm":
        model = LSTMNet()
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

    return model, lr
