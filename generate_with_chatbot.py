import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
import time
import os
import json
import shutil
from datetime import datetime
from openai import OpenAI
import dotenv
from huggingface_hub import hf_hub_download, snapshot_download
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("author_style_api")

# 加载环境变量
dotenv.load_dotenv()

# Author style model API
class AuthorStyleAPI:
    def __init__(self, token=None):
        """
        初始化作者风格 API
        
        参数:
        - token: Hugging Face API token, 默认为 None (将使用 GENERATION_TOKEN 环境变量)
        """
        # 优先使用传入的token，否则从环境变量获取
        self.token = token or os.environ.get("GENERATION_TOKEN")
        
        # 设置模型路径
        self.model_name = "fjxddy/author-stylegan"
        self.local_model_path = "author_style_model"  # 本地模型的根目录
        self.generators_path = os.path.join(self.local_model_path, "generators")  # 生成器模型路径
        
        # 确保本地目录存在
        os.makedirs(self.generators_path, exist_ok=True)
        
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
        self.loaded_models = {}  # 缓存已加载的模型
        self.last_request_time = None
        self.min_request_interval = 2  # 最小请求间隔（秒）
        
        logger.info(f"Initialized AuthorStyleAPI")

    def _wait_for_rate_limit(self):
        """确保请求间隔符合速率限制"""
        if self.last_request_time is not None:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = datetime.now()

    def _download_model(self, author, max_retries=3):
        """
        下载指定作者的模型到本地目录
        
        参数:
        - author: 作者名称
        - max_retries: 最大重试次数
        
        返回:
        - success: 下载是否成功
        """
        if not self.token:
            logger.error("No token available for model download")
            return False
            
        remote_path = f"generators/{author}"
        local_path = os.path.join(self.generators_path, author)
        
        # 如果本地模型已存在，则跳过下载
        if os.path.exists(local_path) and os.path.exists(os.path.join(local_path, "config.json")):
            logger.info(f"Local model already exists at {local_path}")
            return True
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading model for {author} (attempt {attempt+1}/{max_retries})...")
                self._wait_for_rate_limit()
                
                # 创建临时目录
                temp_dir = f"{local_path}_temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                # 下载模型
                snapshot_download(
                    repo_id=self.model_name,
                    local_dir=temp_dir,
                    subfolder=remote_path,
                    token=self.token
                )
                
                # 移动文件到最终目录
                if os.path.exists(local_path):
                    shutil.rmtree(local_path)
                os.makedirs(os.path.dirname(local_path), exist_ok=True)
                shutil.move(temp_dir, local_path)
                
                logger.info(f"Model successfully downloaded to {local_path}")
                return True
            except Exception as e:
                if "429" in str(e):  # 速率限制错误
                    wait_time = (attempt + 1) * 10  # 等待时间指数增长
                    logger.warning(f"API rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error downloading model (attempt {attempt+1}/{max_retries}): {str(e)}")
                    # 清理临时目录
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
                    
                    if attempt == max_retries - 1:
                        logger.error("Failed to download model after multiple retries")
                        return False
        return False
        
    def _load_model(self, author, max_retries=3):
        """加载指定作者的模型，带重试机制"""
        if author not in self.available_authors:
            raise ValueError(f"作者 {author} 不可用。请从以下作者中选择: {', '.join(self.available_authors)}")
            
        if author in self.loaded_models:
            return self.loaded_models[author]

        # 检查本地模型是否存在
        local_author_path = os.path.join(self.generators_path, author)
        local_model_exists = os.path.exists(local_author_path) and os.path.exists(os.path.join(local_author_path, "config.json"))
        
        # 有token且本地不存在，尝试下载
        if self.token and not local_model_exists:
            logger.info(f"Local model not found for {author}, attempting to download")
            download_success = self._download_model(author)
            if download_success:
                logger.info(f"Successfully downloaded model for {author}, using local model")
                local_model_exists = True
            else:
                logger.warning(f"Failed to download model for {author}")
        
        for attempt in range(max_retries):
            try:
                # 优先尝试从本地加载模型
                if local_model_exists:
                    logger.info(f"Loading local model for {author}... (attempt {attempt + 1}/{max_retries})")
                    tokenizer = AutoTokenizer.from_pretrained(local_author_path)
                    model = AutoModelForCausalLM.from_pretrained(local_author_path)
                else:
                    # 如果本地模型不存在，直接从远程加载
                    logger.info(f"Loading remote model for {author}... (attempt {attempt + 1}/{max_retries})")
                    self._wait_for_rate_limit()
                    
                    # 使用 token 如果有的话
                    if self.token:
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            subfolder=f"generators/{author}",
                            token=self.token
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            subfolder=f"generators/{author}",
                            token=self.token
                        )
                    else:
                        # 无 token 时尝试公开下载
                        tokenizer = AutoTokenizer.from_pretrained(
                            self.model_name,
                            subfolder=f"generators/{author}"
                        )
                        model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            subfolder=f"generators/{author}"
                        )
                
                # 确保tokenizer有pad_token
                if tokenizer.pad_token is None:
                    tokenizer.pad_token = tokenizer.eos_token
                    
                self.loaded_models[author] = (tokenizer, model)
                return tokenizer, model
            except Exception as e:
                if "429" in str(e):  # 速率限制错误
                    wait_time = (attempt + 1) * 10  # 等待时间指数增长
                    logger.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error loading model for {author}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise  # 所有重试都失败，抛出异常

    def generate_text(self, author, prompt="", num_samples=1, max_length=200, max_retries=3):
        """
        生成指定作者风格的文本样本，带重试机制
        
        参数:
        - author: 作者名称
        - prompt: 生成文本的提示，默认为空
        - num_samples: 生成样本数量
        - max_length: 生成文本的最大长度
        - max_retries: 最大重试次数
        
        返回:
        - samples: 生成的文本样本列表
        """
        if author not in self.available_authors:
            raise ValueError(f"作者 {author} 不可用。请从以下作者中选择: {', '.join(self.available_authors)}")

        for attempt in range(max_retries):
            try:
                tokenizer, model = self._load_model(author)
                samples = []

                for i in range(num_samples):
                    self._wait_for_rate_limit()
                    logger.info(f"Generating sample {i + 1}/{num_samples}...")
                    
                    # 处理提示文本
                    if prompt:
                        # 如果有提示文本，使用它初始化生成
                        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
                        input_ids = inputs["input_ids"]
                        attention_mask = inputs["attention_mask"]
                    else:
                        # 如果没有提示文本，使用起始标记
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
                if "429" in str(e):  # 速率限制错误
                    wait_time = (attempt + 1) * 10  # 等待时间指数增长
                    logger.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error generating text: {str(e)}")
                    if attempt == max_retries - 1:
                        raise  # 所有重试都失败，抛出异常

# DeepSeek API客户端创建函数
def create_deepseek_client(api_key=None):
    """
    创建DeepSeek API客户端
    
    参数:
    - api_key: API密钥，如不提供则尝试从环境变量获取
    
    返回:
    - client: OpenAI客户端对象
    """
    # 如果没有提供API密钥，尝试从环境变量获取
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
        
    if not api_key:
        raise ValueError("No DeepSeek API key provided and none found in environment variables")
        
    # 创建并返回客户端
    return OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

# DeepSeek conversation function
def chat_with_deepseek(author, prompt="", api_key=None):
    """
    使用DeepSeek API生成指定作者风格的文本
    
    参数:
    - author: 作者名称
    - prompt: 生成文本的提示，默认为空
    - api_key: 可选的API密钥，如不提供则尝试从环境变量获取
    
    返回:
    - text: 生成的文本
    """
    try:
        # 创建DeepSeek客户端
        client = create_deepseek_client(api_key)
        
        # 设置初始对话消息
        conversation = [
            {"role": "system", "content": f"You are a helpful assistant that mimics the style of {author}."}
        ]

        # 请求文本生成
        user_content = prompt if prompt else "Please generate a text sample."
        conversation.append({"role": "user", "content": user_content})

        # 调用DeepSeek API获取响应
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=conversation,
            stream=False
        )

        # 获取DeepSeek回复
        deepseek_reply = response.choices[0].message.content
        logger.info(f"DeepSeek generated text for {author}")

        return deepseek_reply
    except Exception as e:
        logger.error(f"Error when calling DeepSeek API: {str(e)}")
        raise

# 简化的API函数
def generate_text(author, prompt="", num_samples=1, max_length=200, token=None):
    """
    简化的API函数，生成指定作者风格的文本
    
    参数:
    - author: 作者名称
    - prompt: 生成文本的提示，默认为空
    - num_samples: 生成样本数量
    - max_length: 生成文本的最大长度
    - token: Hugging Face API token，默认为None
    
    返回:
    - samples: 生成的文本样本列表
    """
    api = AuthorStyleAPI(token=token)
    return api.generate_text(author, prompt, num_samples, max_length)

# Main program
def main():
    # 从环境变量获取 Hugging Face token
    token = os.environ.get("GENERATION_TOKEN")
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
            # 获取可选的提示文本
            prompt = input("\nEnter a prompt (optional, press Enter to skip): ")
            
            # Generate text using own model
            print(f"\nGenerating text in the style of {author}...")
            samples = api.generate_text(author, prompt)
            print("\nGenerated text:")
            for i, sample in enumerate(samples):
                print(f"\nSample {i+1}:")
                print(sample)

            # Generate text using DeepSeek API
            print("\nDo you want to generate text using DeepSeek API? (y/n)")
            deepseek_choice = input("> ")
            
            if deepseek_choice.lower() == 'y':
                # 检查环境变量中是否有API密钥
                api_key = os.environ.get("OPENAI_API_KEY")
                
                if not api_key:
                    print("\nNo DeepSeek API key found in environment variables.")
                    print("Do you want to enter an API key? (y/n)")
                    key_choice = input("> ")
                    
                    if key_choice.lower() == 'y':
                        api_key = input("Enter your DeepSeek API key: ")
                    else:
                        print("Skipping DeepSeek text generation.")
                        continue
                
                print(f"\nGenerating text in the style of {author} using DeepSeek...")
                deepseek_text = chat_with_deepseek(author, prompt, api_key)
                print("\nDeepSeek generated text:")
                print(deepseek_text)

        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    main()