import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
import time
import os
import json
from datetime import datetime, timedelta
import dotenv

# 加载环境变量
dotenv.load_dotenv()

class AuthorStyleAPI:
    def __init__(self, token):
        self.token = token
        self.model_name = "fjxddy/author-stylegan"
        self.available_authors = [
            "Agatha_Christie",
            "Alexandre_Dumas",
            "Arthur_Conan_Doyle",
            "Charles_Dickens",
            "Charlotte_Brontë",
            "F._Scott_Fitzgerald",
            "García_Márquez",
            "Herman_Melville",
            "Jane_Austen",
            "Mark_Twain"
        ]
        self.loaded_models = {} 
        self.last_request_time = None
        self.min_request_interval = 2 

    def _wait_for_rate_limit(self):

        if self.last_request_time is not None:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = datetime.now()

    def _load_model(self, author, max_retries=3):

        if author in self.loaded_models:
            return self.loaded_models[author]
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                print(f"Loading model for {author}... (attempt {attempt + 1}/{max_retries})")
                tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    subfolder=f"generators/{author}",
                    token=self.token
                )

                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                
                model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    subfolder=f"generators/{author}",
                    token=self.token
                )
                self.loaded_models[author] = (tokenizer, model)
                return tokenizer, model
            except Exception as e:
                if "429" in str(e):  
                    wait_time = (attempt + 1) * 10  
                    print(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error loading model for {author}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise

    def generate_text(self, author, num_samples=1, max_length=200, max_retries=3):

        if author not in self.available_authors:
            raise ValueError(f"Author {author} not available. Please choose from: {', '.join(self.available_authors)}")
        
        for attempt in range(max_retries):
            try:
                tokenizer, model = self._load_model(author)
                samples = []
                
                for i in range(num_samples):
                    self._wait_for_rate_limit()
                    print(f"Generating sample {i + 1}/{num_samples}...")

                    input_ids = torch.tensor([[tokenizer.bos_token_id]])
                    attention_mask = torch.ones_like(input_ids)
                    
                    outputs = model.generate(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        max_length=max_length,
                        temperature=0.9,
                        top_k=40,
                        top_p=0.9,
                        repetition_penalty=1.2,
                        do_sample=True,
                        pad_token_id=tokenizer.pad_token_id
                    )
                    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                    samples.append(generated_text)
                
                return samples
            except Exception as e:
                if "429" in str(e):  
                    wait_time = (attempt + 1) * 10  
                    print(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error generating text: {str(e)}")
                    if attempt == max_retries - 1:
                        raise

    def evaluate_text(self, text, author, max_retries=3):

        if author not in self.available_authors:
            raise ValueError(f"Author {author} not available. Please choose from: {', '.join(self.available_authors)}")
        
        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()

                from transformers import BertForSequenceClassification, BertTokenizer
                

                tokenizer = BertTokenizer.from_pretrained(
                    self.model_name,
                    subfolder=f"discriminators/{author}/best_model",
                    token=self.token
                )
                model = BertForSequenceClassification.from_pretrained(
                    self.model_name,
                    subfolder=f"discriminators/{author}/best_model",
                    token=self.token
                )
                

                label_path = f"discriminators/{author}/best_model/label_names.json"
                try:
                    from huggingface_hub import hf_hub_download
                    label_file = hf_hub_download(
                        repo_id=self.model_name,
                        filename=label_path,
                        token=self.token
                    )
                    with open(label_file, "r") as f:
                        author_labels = json.load(f)
                except Exception as e:
                    print(f"Warning: Could not load label names: {str(e)}")
                    author_labels = [None, author]  
                

                author_indices = {author: idx for idx, author in enumerate(author_labels) if author is not None}
                author_idx = author_indices.get(author, 1)  
                

                inputs = tokenizer(
                    text,
                    return_tensors="pt",
                    padding='max_length',
                    truncation=True,
                    max_length=512
                )
                

                with torch.no_grad():
                    outputs = model(**inputs)
                    logits = outputs.logits
                    probs = F.softmax(logits, dim=1)
                    score = probs[0][author_idx].item()
                
                return score
            except Exception as e:
                if "429" in str(e):  
                    wait_time = (attempt + 1) * 10  
                    print(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    print(f"Error evaluating text: {str(e)}")
                    if attempt == max_retries - 1:
                        raise

    def generate_best_sample(self, author, num_samples=10, max_length=200):
        """生成多个样本并返回评分最高的一个"""
        samples = self.generate_text(author, num_samples, max_length)
        best_sample = None
        best_score = -1
        
        for i, sample in enumerate(samples):
            try:
                print(f"Evaluating sample {i + 1}/{len(samples)}...")
                score = self.evaluate_text(sample, author)
                if score > best_score:
                    best_score = score
                    best_sample = sample
            except Exception as e:
                print(f"Error evaluating sample: {str(e)}")
                continue
        
        return best_sample, best_score

def main():

    # 从环境变量获取 Hugging Face token
    token = os.environ.get("HUGGINGFACE_TOKEN")
    # 确保 token 已设置
    if not token:
        raise ValueError("请确保在 .env 文件中设置了 HUGGINGFACE_TOKEN 环境变量")
        
    api = AuthorStyleAPI(token)
    
    print("Available authors:")
    for author in api.available_authors:
        print(f"- {author}")
    
    while True:
        author = input("\nEnter author name (or 'quit' to exit): ")
        if author.lower() == 'quit':
            break
            
        if author not in api.available_authors:
            print(f"Author {author} not available. Please choose from the list above.")
            continue
            
        try:
            print(f"\nGenerating text in the style of {author}...")
            best_sample, score = api.generate_best_sample(author)
            print("\nBest generated text:")
            print(best_sample)
            print(f"\nStyle match score: {score:.4f}")
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()