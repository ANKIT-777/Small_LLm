import json
import pickle
import torch
from pathlib import Path
from config import config

class TextProcessor:
    def __init__(self):
        self.vocab = None
        self.char_to_idx = None
        self.idx_to_char = None
        
    def build_vocab(self, text):
        # Add special tokens and handle unknown chars
        chars = sorted(list(set(text)))
        chars = ['<UNK>', '<PAD>'] + chars
        self.char_to_idx = {ch: i for i, ch in enumerate(chars)}
        self.idx_to_char = {i: ch for i, ch in enumerate(chars)}
        self.vocab = chars
        
    def save_vocab(self):
        with open(config.VOCAB_FILE, 'wb') as f:
            pickle.dump((self.char_to_idx, self.idx_to_char), f)
            
    def load_vocab(self):
        with open(config.VOCAB_FILE, 'rb') as f:
            self.char_to_idx, self.idx_to_char = pickle.load(f)
        self.vocab = list(self.char_to_idx.keys())
        
    def text_to_tensor(self, text):
        # Handle unknown characters with <UNK> token
        indices = [self.char_to_idx.get(ch, self.char_to_idx['<UNK>']) for ch in text]
        return torch.tensor(indices, dtype=torch.long)
    
    def tensor_to_text(self, tensor):
        return ''.join([self.idx_to_char[idx.item()] for idx in tensor])
    
    def prepare_data(self):
        # Load resume text
        with open(config.RESUME_FILE, 'r') as f:
            resume_text = f.read()
        
        # Load JSON data
        with open(config.JSON_FILE, 'r') as f:
            json_data = json.load(f)
        
        # Convert JSON to formatted text
        json_text = self.json_to_text(json_data)
        
        # Combine both texts
        combined_text = resume_text + "\n" + json_text
        
        self.build_vocab(combined_text)
        self.save_vocab()
        
        # Load QA pairs if available
        qa_pairs = []
        if config.QA_FILE.exists():
            with open(config.QA_FILE, 'r') as f:
                qa_pairs = json.load(f)
        
        return combined_text, qa_pairs

    def json_to_text(self, data):
        sections = []
        
        # Personal Info
        sections.append("[PERSONAL]")
        for key, value in data['personal_info'].items():
            if isinstance(value, list):
                sections.append(f"- {key}: {', '.join(value)}")
            else:
                sections.append(f"- {key}: {value}")
        
        # Skills
        sections.append("\n[DETAILED_SKILLS]")
        for category, skills in data['skills'].items():
            if isinstance(skills, list):
                sections.append(f"- {category}: {', '.join(skills)}")
            else:
                sections.append(f"- {category}: {skills}")
        
        # Experience
        sections.append("\n[DETAILED_EXPERIENCE]")
        for job in data['experience']:
            sections.append(f"- {job['title']} at {job['company']}")
            sections.append(f"  Duration: {job['duration']}")
            sections.append(f"  Location: {job['location']}")
            for resp in job['responsibilities']:
                sections.append(f"  * {resp}")
        
        # Projects
        sections.append("\n[DETAILED_PROJECTS]")
        for project in data['projects']:
            sections.append(f"- {project['title']}")
            sections.append(f"  Technologies: {', '.join(project['technologies'])}")
            sections.append(f"  Description: {project['description']}")
            sections.append(f"  Impact: {project['impact']}")
        
        return "\n".join(sections)
