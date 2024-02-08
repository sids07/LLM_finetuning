from decoding_strategies.config import MODEL_NAME
from transformers import GPT2LMHeadModel
from transformers import GPT2Tokenizer

class DecodingInference:
    
    def __init__(self):
        
        self.tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
        self.model = GPT2LMHeadModel.from_pretrained(MODEL_NAME)
        
    def generate(self,prompt, **kwargs):
        input_ids = self.tokenizer.encode(prompt + self.tokenizer.eos_token, return_tensors="pt")
        
        history_ids = self.model.generate(input_ids, **kwargs)
        output_list = []
        for output in history_ids:
            output_list.append(
                self.tokenizer.decode(
                    output[input_ids.shape[-1]:], skip_special_tokens=True
                    )
                )
        return output_list[0]  
    
    def generation_using_different_strategies(self, prompt, strategy_mode="greedy_search"):
        
        if strategy_mode == "greedy_search":
            kwargs = {}
        
        elif strategy_mode == "beam_search":
            kwargs = {
                "num_beams":5,
                "early_stopping":True,
                "no_repeat_ngram_size":2,
                "num_return_sequences":5,
                "max_length":50
            }
            
        elif strategy_mode == "random":
            kwargs = {
                "temperature":0.7, 
                "do_sample":True, 
                "top_k":0, 
                "max_length":50
            }
        
        elif strategy_mode == "top_k":
            kwargs = {
                "temperature":0.7, 
                "do_sample":True, 
                "top_k":50, 
                "max_length":50
            }
        
        elif strategy_mode == "top_p":
            kwargs = {
                "do_sample":True, 
                "top_p":0.9,
                "top_k":50, 
                "max_length":50
            }
        
        else:
            raise ValueError("Please provide strategy_mode from ['greedy_search','beam_search','random','top_k','top_p']")
        
        response = self.generate(
                prompt=prompt,
                **kwargs
            )
        
        return response

if __name__ == "__main__":
    decoding_inference = DecodingInference()
    
    while True:
        decoding_mode = print("Enter what decoding strategies you want to use: ")
        
        prompt = print("Enter your query: ")
        
        response = decoding_inference.generation_using_different_strategies(
            prompt= prompt,
            strategy_mode= decoding_mode
        )
        
        print(response)