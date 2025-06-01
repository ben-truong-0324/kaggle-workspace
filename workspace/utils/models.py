# scikit-learn classifiers (for multinomial_classification task_type)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# scikit-learn regressors (for prob_vector and regression task_type)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression # For LogisticRegression's regressor equivalent if needed
from sklearn.svm import SVR # Support Vector Regressor
from sklearn.neighbors import KNeighborsRegressor

# MultiOutput wrapper
from sklearn.multioutput import MultiOutputRegressor

# gradient boosting frameworks
from xgboost import XGBClassifier, XGBRegressor # Import both
from lightgbm import LGBMClassifier, LGBMRegressor # Import both


def get_sklearn_model(name: str, task_type: str, **kwargs):
    if task_type == "multinomial_classification":
        # Classifiers for discrete class prediction
        if name == "decision_tree":
            return DecisionTreeClassifier(**kwargs)
        elif name == "random_forest":
            return RandomForestClassifier(**kwargs)
        elif name == "logistic_regression":
            # LogisticRegression is inherently a classifier. For multi-output
            # continuous data, its regression equivalent is typically handled
            # by LinearRegression wrapped in MultiOutputRegressor or not used.
            # If you specifically need a classifier for this name, keep it here.
            return LogisticRegression(**kwargs)
        elif name == "svm":
            return SVC(**kwargs)
        elif name == "naive_bayes":
            return GaussianNB(**kwargs)
        elif name == "knn":
            return KNeighborsClassifier(**kwargs)
        elif name == "xgboost":
            # For XGBoost classifier, ensure use_label_encoder=False for newer versions
            return XGBClassifier(use_label_encoder=False, eval_metric="logloss", **kwargs)
        elif name == "lightgbm":
            return LGBMClassifier(**kwargs)
        else:
            raise ValueError(f"Unknown classifier model for 'multinomial_classification': {name}")

    elif task_type == "prob_vector" or task_type == "regression":
        # For 'prob_vector' (multi-output regression) or standard 'regression' (single-output)
        # We need regressors, and for 'prob_vector', they should be wrapped in MultiOutputRegressor.

        base_model = None

        if name == "decision_tree":
            base_model = DecisionTreeRegressor(**kwargs)
        elif name == "random_forest":
            base_model = RandomForestRegressor(**kwargs)
        elif name == "logistic_regression":
            # For 'prob_vector' or 'regression', LogisticRegression is not directly applicable.
            # You might use LinearRegression as a simple base.
            base_model = LinearRegression(**kwargs)
        elif name == "svm":
            base_model = SVR(**kwargs)
        elif name == "knn":
            base_model = KNeighborsRegressor(**kwargs)
        elif name == "xgboost":
            base_model = XGBRegressor(objective='reg:squarederror', **kwargs) # Standard regression objective
        elif name == "lightgbm":
            base_model = LGBMRegressor(**kwargs)
        else:
            raise ValueError(f"Unknown regressor model for '{task_type}' task: {name}")

        # Wrap in MultiOutputRegressor if the task is 'prob_vector'
        if task_type == "prob_vector":
            return MultiOutputRegressor(estimator=base_model)
        else: # This is for standard 'regression' (single output)
            return base_model

    else:
        raise ValueError(f"Unknown task_type: {task_type}")
    
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
            # layers.append(nn.Softmax(dim=1))
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
                nn.Linear(32, output_dim),
                # nn.Softmax(dim=1)
            )

        def forward(self, x):
            # Ensure x has shape (batch_size, 1, input_dim) for Conv1d
            if x.dim() == 2: # if input is (batch_size, input_dim)
                 x = x.unsqueeze(1) # Add channel dimension
            x = self.conv(x).squeeze(-1) # Squeeze the last dimension (from AdaptiveAvgPool1d)
            return self.fc(x)
        
    class LSTMNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, batch_first=True)
            self.fc = nn.Sequential(
                nn.Linear(hidden, output_dim),
                # nn.Softmax(dim=1) 
            )


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





def get_custom_nn_model(input_dim: int, output_dim: int = 1, **kwargs):
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
  
    # Define the dropout layer
    dropout_layer = nn.Dropout(dropout) if dropout > 0 else None
  
    # === MLP Custom Model ===
    class MLP_custom(nn.Module):
        def __init__(self):
            super().__init__()
            layers = []
            
            # Input Layer to First Hidden, either expand or compress depending on hyperparam
            layers.append(nn.Linear(input_dim, hidden))
            layers.append(nn.Sigmoid())  #sigmoid to [0,1]
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
            
            # expand hidden state
            layers.append(nn.Linear(hidden, hidden*2))
            layers.append(nn.Tanh())  #tanh, to strengthen or weaken nodes at expansion
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden*2))

            layers.append(nn.Linear(hidden*2, hidden))
            layers.append(nn.ReLU())   #relu, non-linearity here at compression
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden))
                
            if dropout > 0:
                layers.append(dropout_layer)
            layers.append(nn.Linear(hidden, output_dim))
            self.net = nn.Sequential(*layers)
        
        def forward(self, x):
            return self.net(x)
  
    # === LSTM Custom Model ===
    class LSTMNet_custom(nn.Module):
        def __init__(self):
            super().__init__()
            self.lstm = nn.LSTM(input_size=input_dim, hidden_size=hidden, batch_first=True)
            self.dropout_layer = dropout_layer if dropout > 0 else None
            self.fc = nn.Sequential(
                nn.Linear(hidden, output_dim),
            )
        
        def forward(self, x):
            # Add sequence dimension if not present (assuming input is [batch, features])
            if len(x.shape) == 2:
                x = x.unsqueeze(1)
            
            # LSTM Forward Pass
            out, (hn, cn) = self.lstm(x)
            
            # Apply dropout if applicable
            if self.dropout_layer is not None:
                out = F.dropout(out, p=dropout, training=self.training)
            # Final FC Layer
            out = self.fc(hn[-1])  # Using the last hidden state
            
            return out
  
    # === Attention Mechanism Helper Class ===
    class Attention(nn.Module):
        def __init__(self, input_dim, output_dim):
            super().__init__()
            self.attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=2)
            self.fc = nn.Linear(input_dim, output_dim)
        
        def forward(self, x):
            # Reshape for attention
            seq_len = x.size(1)
            x_reshaped = x.view(-1, seq_len, x.size(-1))
            
            # Apply attention
            attn_output, _ = self.attention(x_reshaped, x_reshaped, x_reshaped)
            
            # Average over sequence length
            attn_avg = attn_output.mean(dim=1)
            
            # Final FC Layer
            out = self.fc(attn_avg)
            
            return out
  
    # === Get Model and Optimizer ===
    if model_type == "mlp":
        model = MLP_custom()
    elif model_type == "lstm":
        model = LSTMNet_custom()
    elif model_type == "attention":
        # Assuming input_dim is the feature dimension for attention
        model = Attention(input_dim=input_dim, output_dim=output_dim)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
  
    # Return the model and learning rate
    return model, lr