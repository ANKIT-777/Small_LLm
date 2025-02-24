import torch
from models.gpt_j_wrapper import GPTWrapper
from models.resume_llm import ResumeLLM
from utils.preprocess import TextProcessor
from config import config

class HybridGenerator:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.processor = TextProcessor()
        self.processor.load_vocab()
        
        # Load custom LLM
        self.resume_model = ResumeLLM(
            vocab_size=len(self.processor.vocab),
            emb_dim=config.EMBED_DIM,
            hidden_dim=config.HIDDEN_DIM,
            n_layers=config.N_LAYERS,
            dropout=config.DROPOUT
        ).to(self.device)
        self.resume_model.load_state_dict(torch.load(config.MODEL_FILE))
        self.resume_model.eval()
        
        # Load GPT-2 instead of GPT-J
        self.gpt_j = GPTWrapper()
        
    def generate(self, prompt, use_gpt_j=False):
        if use_gpt_j:
            return self.generate_gpt_j(prompt)
        return self.generate_resume(prompt)
    
    def generate_resume(self, prompt):
        with torch.no_grad():
            # Clean the prompt
            prompt = prompt.upper() if prompt.isupper() else prompt.lower()
            if not prompt.startswith('['):
                prompt = f"[{prompt}]"
            
            input_seq = self.processor.text_to_tensor(prompt).to(self.device)
            hidden = self.resume_model.init_hidden(1)
            hidden = (hidden[0].to(self.device), hidden[1].to(self.device))
            
            # Start with empty response
            generated = []
            found_content = False
            
            for _ in range(config.MAX_LENGTH):
                output, hidden = self.resume_model(input_seq.unsqueeze(0), hidden)
                probs = torch.softmax(output[0, -1] / config.TEMPERATURE, dim=-1)
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
    
    def generate_gpt_j(self, prompt):
        # Load resume context from file
        try:
            with open(config.RESUME_FILE, 'r') as f:
                resume_context = f.read()
        except FileNotFoundError:
            resume_context = ""
        
        return self.gpt_j.generate_response(prompt, resume_context)

if __name__ == "__main__":
    generator = HybridGenerator()
    print("\nExample prompts: SKILLS, EXPERIENCE, EDUCATION, PROJECTS")
    print("Add '!gpt' to use GPT-J for more detailed responses")
    print("Type 'exit' or 'quit' to end\n")
    
    while True:
        prompt = input("\nAsk about the resume: ")
        if prompt.lower() in ['exit', 'quit']:
            break
            
        use_gpt_j = '!gpt' in prompt.lower()
        prompt = prompt.replace('!gpt', '').strip()
        
        response = generator.generate(prompt, use_gpt_j=use_gpt_j)
        print(f"Response: {response}")