import os
from pathlib import Path

class Config:
    # Paths
    BASE_DIR = Path(__file__).parent
    DATA_DIR = BASE_DIR / "data"
    MODEL_DIR = BASE_DIR / "models"
    
    # Training
    BATCH_SIZE = 32
    SEQ_LENGTH = 50
    EMBED_DIM = 64
    HIDDEN_DIM = 128
    N_LAYERS = 1
    DROPOUT = 0.0
    LEARNING_RATE = 0.001
    EPOCHS = 500
    
    # Inference
    TEMPERATURE = 0.5
    MAX_LENGTH = 300
    
    # Data files
    RESUME_FILE = DATA_DIR / "resume.txt"
    JSON_FILE = DATA_DIR / "jsondata.json"
    QA_FILE = DATA_DIR / "qa_pairs.json"
    
    # Model files
    MODEL_FILE = MODEL_DIR / "resume_llm.pth"
    VOCAB_FILE = MODEL_DIR / "vocab.pkl"

config = Config()