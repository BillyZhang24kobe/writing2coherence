import torch 
import torch.nn as nn


class Pooler(nn.Module):
    def __init__(self, config, device: str= None):
        super().__init__()
        self.device = device
        self.dense = nn.Linear(config.hidden_size, config.hidden_size).to(self.device)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token.
        first_token_tensor = hidden_states[:, 0]
        pooled_output = self.dense(first_token_tensor).to(self.device)
        pooled_output = self.activation(pooled_output)
        return pooled_output.to(self.device)

