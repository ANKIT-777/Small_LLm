from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class GPTWrapper:
    def __init__(self):
        model_name = "distilgpt2"  # Even smaller model (82M parameters)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        
    def generate_response(self, prompt, resume_context):
        # Format the prompt with resume context
        full_prompt = f"""
        Resume Information:
        {resume_context}
        
        Question: {prompt}
        Answer:"""
        
        inputs = self.tokenizer(full_prompt, return_tensors="pt", truncation=True)
        outputs = self.model.generate(
            inputs["input_ids"],
            max_length=500,          # Increased from 150
            min_length=100,          # Added minimum length
            temperature=0.8,         # Slightly increased for more variety
            num_return_sequences=1,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            top_p=0.95,             # Added top_p sampling
            repetition_penalty=1.2   # Added to reduce repetition
        )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True) 