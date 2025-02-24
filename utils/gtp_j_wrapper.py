from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GPTJWrapper:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Loading GPT-J on {self.device}...")
        
        # Load model and tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")
        self.model = AutoModelForCausalLM.from_pretrained(
            "EleutherAI/gpt-j-6B",
            torch_dtype=torch.float16,  # Use half precision to save memory
            low_cpu_mem_usage=True
        ).to(self.device)
