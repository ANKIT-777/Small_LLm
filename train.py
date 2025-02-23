import torch
from tqdm import tqdm
from models.resume_llm import ResumeLLM
from utils.preprocess import TextProcessor
from utils.train_utils import create_data_loader, train_epoch
from config import config

def main():
    # Set device (CPU or GPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize components
    processor = TextProcessor()
    resume_text, _ = processor.prepare_data()
    
    # Create model and move to device
    model = ResumeLLM(
        vocab_size=len(processor.vocab),
        emb_dim=config.EMBED_DIM,
        hidden_dim=config.HIDDEN_DIM,
        n_layers=config.N_LAYERS,
        dropout=config.DROPOUT
    ).to(device)
    
    # Training setup
    data_loader = create_data_loader(resume_text, config.BATCH_SIZE, config.SEQ_LENGTH, processor)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.LEARNING_RATE)
    
    # Training loop with progress bar
    for epoch in tqdm(range(config.EPOCHS), desc="Training"):
        loss = train_epoch(model, data_loader, criterion, optimizer, device)
        if epoch % 10 == 0:
            print(f"Epoch {epoch} | Loss: {loss:.4f}")
            
    # Save model
    torch.save(model.state_dict(), config.MODEL_FILE)
    print(f"Model saved to {config.MODEL_FILE}")

if __name__ == "__main__":
    main()