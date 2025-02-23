import torch
import torch.nn as nn
import torch.nn.functional as F

class ResumeLLM(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, n_layers, dropout):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.lstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden):
        # x shape: (batch_size, seq_length)
        embedded = self.embedding(x)  # (batch_size, seq_length, emb_dim)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)  # (batch_size, seq_length, vocab_size)
        return output, hidden
    
    def init_hidden(self, batch_size):
        # Initialize hidden state and cell state
        # Shape: (num_layers, batch_size, hidden_dim)
        h0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        c0 = torch.zeros(self.n_layers, batch_size, self.hidden_dim)
        return (h0, c0)