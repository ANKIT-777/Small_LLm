import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    
    # Training
    BATCH_SIZE = 64
    SEQ_LENGTH = 100
    EMBED_DIM = 256
    HIDDEN_DIM = 512
    N_LAYERS = 2
    DROPOUT = 0.2
    LEARNING_RATE = 0.001
    EPOCHS = 5000
    
    # Inference
    TEMPERATURE = 0.7
    MAX_LENGTH = 200
    
    # Data files
    RESUME_FILE = DATA_DIR / "resume.txt"
    QA_FILE = DATA_DIR / "qa_pairs.json"
    
    # Model files
    MODEL_FILE = MODEL_DIR / "resume_llm.pth"
    VOCAB_FILE = MODEL_DIR / "vocab.pkl"

config = Config()