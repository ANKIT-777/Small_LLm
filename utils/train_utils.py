import torch
from torch.utils.data import Dataset, DataLoader
from config import config
from utils.preprocess import TextProcessor

class ResumeDataset(Dataset):
    def __init__(self, text, seq_length, processor):
        self.text = text
        self.seq_length = seq_length
        self.processor = processor
        
    def __len__(self):
        return len(self.text) - self.seq_length
    
    def __getitem__(self, idx):
        # Ensure sequences are within bounds
        if idx + self.seq_length >= len(self.text):
            input_seq = self.text[idx:] + '<PAD>' * (self.seq_length - (len(self.text) - idx))
            target_seq = self.text[idx+1:] + '<PAD>' * (self.seq_length - (len(self.text) - idx) + 1)
        else:
            input_seq = self.text[idx:idx+self.seq_length]
            target_seq = self.text[idx+1:idx+self.seq_length+1]
        
        return (
            self.processor.text_to_tensor(input_seq),
            self.processor.text_to_tensor(target_seq)
        )


def create_data_loader(text, batch_size, seq_length, processor):
    dataset = ResumeDataset(text, seq_length, processor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

def train_epoch(model, data_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    
    for batch_idx, (inputs, targets) in enumerate(data_loader):
        inputs = inputs.to(device)
        targets = targets.to(device)
        
        hidden = model.init_hidden(inputs.size(0))
        hidden = (hidden[0].to(device), hidden[1].to(device))
        
        optimizer.zero_grad()
        outputs, hidden = model(inputs, hidden)
        
        loss = criterion(outputs.view(-1, outputs.size(-1)), targets.view(-1))
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item()
        
        if batch_idx % 10 == 0:  # Print progress more frequently
            print(f"\rBatch {batch_idx}/{len(data_loader)} | Loss: {loss.item():.4f}", end="")
            
    return total_loss / len(data_loader)

