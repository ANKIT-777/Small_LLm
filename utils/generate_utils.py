import torch
from config import config

class ResumeGenerator:
    def __init__(self, model, processor):
        self.model = model
        self.processor = processor
        
    def generate(self, prompt, max_length=config.MAX_LENGTH, temperature=config.TEMPERATURE):
        self.model.eval()
        with torch.no_grad():
            input_seq = self.processor.text_to_tensor(prompt)
            hidden = None
            generated = []
            
            for _ in range(max_length):
                output, hidden = self.model(input_seq.unsqueeze(0), hidden)
                probs = torch.softmax(output[-1] / temperature, dim=0)
                next_idx = torch.multinomial(probs, 1)
                next_char = self.processor.idx_to_char[next_idx.item()]
                
                if next_char == "<":  # Stop at section marker
                    break
                    
                generated.append(next_char)
                input_seq = next_idx
                
            return ''.join(generated)