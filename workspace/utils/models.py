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

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


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




# === Positional Encoding Helper Class ===
class PositionalEncoding(nn.Module):
    """
    Injects some information about the relative or absolute position of the tokens in the sequence.
    The positional encodings have the same dimension as the embeddings so that the two can be summed.
    Here, we use sine and cosine functions of different frequencies.
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create a long enough P matrix that can be sliced
        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)) # (d_model/2)
        
        pe = torch.zeros(max_len, 1, d_model) # (max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # register_buffer makes it part of the model's state_dict, but not a parameter

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, embedding_dim] if batch_first=True for Transformer
               or [seq_len, batch_size, embedding_dim] if batch_first=False.
               For this PositionalEncoding, it assumes x is already [seq_len, batch_size, embedding_dim]
               or it's applied after permutation if input is batch_first.
               However, nn.TransformerEncoderLayer with batch_first=True expects [batch, seq, feature].
               So, this PE layer should also expect [batch, seq, feature] if used before batch_first Transformer.
               Let's adjust to expect x: [batch_size, seq_len, d_model]
        """
        # x is [batch_size, seq_len, d_model]
        # self.pe is [max_len, 1, d_model]. We need to slice it for current seq_len
        # and make it [1, seq_len, d_model] to broadcast across batch.
        # Or, more commonly, self.pe is [max_len, d_model] and then unsqueezed.
        # Let's use pe of shape [max_len, d_model] and then slice & unsqueeze.
        # Re-defining pe storage for [max_len, d_model] for easier slicing.
        # If pe is [max_len, 1, d_model], then pe[:x.size(1)] is [seq_len, 1, d_model]
        # To add to x [batch_size, seq_len, d_model], we need to permute pe.
        # x = x + self.pe[:x.size(1)].permute(1,0,2) # if pe is [max_len, 1, d_model] and x is [batch, seq, d_model]
        
        # Simpler: if pe is [max_len, d_model]
        # pe_for_batch = self.pe[:x.size(1), :].unsqueeze(0) # shape [1, seq_len, d_model]
        # x = x + pe_for_batch 
        
        # Current self.pe is [max_len, 1, d_model]. Transpose to [max_len, d_model, 1] then squeeze.
        # Or directly use it by permuting:
        # x is [batch_size, seq_len, d_model]
        # self.pe[:x.size(1)] is [seq_len, 1, d_model]. This needs to be broadcasted.
        # The 1 in dim 1 of pe will broadcast to batch_size.
        # So, self.pe[:x.size(1)].permute(1,0,2) gives [1, seq_len, d_model]
        x = x + self.pe[:x.size(1), :].permute(1, 0, 2).detach() # Use detach if PE is not learned
        return self.dropout(x)

class PositionalEncodingCorrected(nn.Module): # Corrected version for [batch, seq, feature]
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1) # (max_len, 1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)) # (d_model/2)
        
        pe = torch.zeros(max_len, d_model) # (max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe is now [max_len, d_model]
        self.register_buffer('pe', pe.unsqueeze(0)) # Shape [1, max_len, d_model] for easy broadcasting

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor, shape [batch_size, seq_len, d_model]
        """
        # self.pe is [1, max_len, d_model]
        # We need [1, seq_len, d_model]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

##########

# === Transformer Classifier Model ===
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, d_model: int, nhead: int,
                 num_encoder_layers: int, dim_feedforward: int, dropout: float,
                 activation_name: str = "relu"):
        super().__init__()
        self.d_model = d_model
        
        # Input projection: from input_dim (features per timestep) to d_model
        self.input_fc = nn.Linear(input_dim, d_model)
        
        # Positional Encoding
        self.pos_encoder = PositionalEncodingCorrected(d_model, dropout) # Using corrected version
        
        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation_name, # 'relu' or 'gelu' or a callable
            batch_first=True  # Crucial: input format (batch, seq, feature)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_encoder_layers
        )
        
        # Output layer for classification
        # Takes the mean of the sequence output from the transformer
        self.output_fc = nn.Linear(d_model, output_dim)

    def forward(self, src: torch.Tensor) -> torch.Tensor:
        """
        Args:
            src: Input tensor, shape [batch_size, seq_len, input_dim]
        Returns:
            Output tensor, shape [batch_size, output_dim] (logits)
        """
        # Project input features to d_model
        src = self.input_fc(src)  # Shape: [batch_size, seq_len, d_model]
        
        # Add positional encoding
        src = self.pos_encoder(src) # Shape: [batch_size, seq_len, d_model]
        
        # Pass through Transformer encoder
        # src_mask and src_key_padding_mask can be added if needed
        output = self.transformer_encoder(src)  # Shape: [batch_size, seq_len, d_model]
        
        # Global average pooling over the sequence dimension
        # This aggregates information across all time steps
        output = output.mean(dim=1)  # Shape: [batch_size, d_model]
        
        # Final classification layer
        output = self.output_fc(output)  # Shape: [batch_size, output_dim]
        
        return output




def get_custom_nn_model(input_dim: int, output_dim: int = 1, model_type = "mlp", **kwargs):
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

    # === Attention Mechanism Helper Class (User's original) ===
    class Attention(nn.Module):
        def __init__(self, attn_input_dim, attn_output_dim): # Renamed params to avoid confusion
            super().__init__()
            # nn.MultiheadAttention expects embed_dim as the feature dimension
            # For this simple attention, let's assume attn_input_dim is the feature dim (E)
            # And input x is (N, L, E) i.e. (batch, seq_len, feature_dim)
            # If input_dim from get_custom_nn_model is feature_dim, then attn_input_dim = input_dim
            self.attention = nn.MultiheadAttention(embed_dim=attn_input_dim, num_heads=2, batch_first=True)
            self.fc = nn.Linear(attn_input_dim, attn_output_dim)
        
        def forward(self, x):
            # Assuming x is [batch_size, seq_len, feature_dim (attn_input_dim)]
            if len(x.shape) == 2: # If [batch, features], assume seq_len = 1
                x = x.unsqueeze(1) # [batch_size, 1, feature_dim]

            # Apply attention
            # Query, Key, Value are all x
            attn_output, _ = self.attention(x, x, x) # attn_output shape [batch, seq_len, embed_dim]
            
            # Average over sequence length
            attn_avg = attn_output.mean(dim=1) # [batch, embed_dim]
            
            # Final FC Layer
            out = self.fc(attn_avg) # [batch, output_dim]
            
            return out
  
  
    # === Get Model and Optimizer ===
    if model_type == "mlp":
        model = MLP_custom()
    elif model_type == "lstm":
        model = LSTMNet_custom()
    elif model_type == "attention":
        # Assuming input_dim is the feature dimension for attention
        model = Attention(attn_input_dim=input_dim, attn_output_dim=output_dim)

    elif model_type == "transformer":
        # Transformer specific hyperparameters
        d_model = kwargs.get("d_model", 128) # Internal dimension of the transformer
        nhead = kwargs.get("nhead", 4)       # Number of attention heads
        # num_encoder_layers uses 'num_layers' from kwargs, defaulting to 2
        dim_feedforward = kwargs.get("dim_feedforward", 256) # Dimension of feedforward network

        model = TransformerClassifier(
            input_dim=input_dim, # Features per time step
            output_dim=output_dim,
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers, # Reusing num_layers for transformer encoder layers
            dim_feedforward=dim_feedforward,
            dropout=dropout_rate,
            activation_name=activation_name # 'relu' or 'gelu'
        )
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
  
    # Return the model and learning rate
    return model, lr