import torch
from models.resume_llm import ResumeLLM
from utils.preprocess import TextProcessor
from config import config

class ResumeGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TextProcessor()
        self.processor.load_vocab()
    
        self.model = ResumeLLM(
            vocab_size=len(self.processor.vocab),
            emb_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_layers=config.N_LAYERS,
            
            dropout=config.DROPOUT
        ).to(self.device)
        self.model.load_state_dict(torch.load(config.MODEL_FILE))
        self.model.eval()
        
    def generate(self, prompt, max_length=config.MAX_LENGTH, temperature=config.TEMPERATURE):
        with torch.no_grad():
            # Clean the prompt
            prompt = prompt.upper() if prompt.isupper() else prompt.lower()
            if not prompt.startswith('['):
                prompt = f"[{prompt}]"
            
            input_seq = self.processor.text_to_tensor(prompt).to(self.device)
            hidden = self.model.init_hidden(1)
            hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
            
            # Start with empty response
            generated = []
            found_content = False
            
            for _ in range(max_length):
                output, hidden = self.model(input_seq.unsqueeze(0), hidden)
                probs = torch.softmax(output[0, -1] / temperature, dim=-1)
                next_idx = torch.multinomial(probs, 1)[0]
                next_char = self.processor.idx_to_char[next_idx.item()]
                
                # Start collecting after finding the section
                if next_char == '[':
                    found_content = True
                    continue
                
                if found_content and next_char == '[':
                    break
                
                if found_content:
                    generated.append(next_char)
                
                input_seq = torch.tensor([next_idx], dtype=torch.long).to(self.device)
            
            return ''.join(generated).strip()

if __name__ == "__main__":
    generator = ResumeGenerator()
    print("\nExample prompts: SKILLS, EXPERIENCE, EDUCATION, PROJECTS")
    print("Type 'exit' or 'quit' to end\n")
    
    while True:
        prompt = input("\nAsk about the resume: ")
        if prompt.lower() in ['exit', 'quit']:
            break
        response = generator.generate(prompt)
        print(f"Response: {response}")