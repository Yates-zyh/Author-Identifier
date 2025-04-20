import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer, BertForSequenceClassification, BertTokenizer
import time
import os
import json
import shutil
import subprocess
from datetime import datetime
from openai import OpenAI
import dotenv
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
        self.discriminators_path = os.path.join(self.local_model_path, "discriminators")  # 鉴别器模型路径
        
        # 确保本地目录存在
        os.makedirs(self.generators_path, exist_ok=True)
        os.makedirs(self.discriminators_path, exist_ok=True)
        
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

    def _download_model_with_cli(self, remote_path, local_path, token=None, recursive=True):
        """
        使用huggingface-cli命令下载模型
        
        参数:
        - remote_path: 远程模型路径
        - local_path: 本地存储路径
        - token: Hugging Face API token，可选参数
        - recursive: 是否递归下载整个目录
        
        返回:
        - success: 下载是否成功
        """
        try:
            # 创建临时目录用于存储下载的文件
            temp_dir = f"{local_path}_temp"
            os.makedirs(temp_dir, exist_ok=True)
            
            # 构建下载命令
            pattern = f"{remote_path}/**" if recursive else remote_path
            
            # 如果有token才添加token参数
            if token:
                cmd = [
                    "huggingface-cli", "download",
                    "--token", token,
                    "--include", pattern,
                    "--local-dir", temp_dir,
                    self.model_name
                ]
            else:
                cmd = [
                    "huggingface-cli", "download",
                    "--include", pattern,
                    "--local-dir", temp_dir,
                    self.model_name
                ]
            
            logger.info(f"Downloading from {remote_path} using huggingface-cli...")
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode != 0:
                logger.warning(f"huggingface-cli download failed: {result.stderr}")
                # 清理临时目录
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                return False
            
            # 创建目标目录
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 如果目标目录已存在，先删除
            if os.path.exists(local_path):
                shutil.rmtree(local_path)
                
            # 移动文件到最终位置
            # 第一步：获取下载的目录路径
            downloaded_path = os.path.join(temp_dir, remote_path)
            
            if os.path.exists(downloaded_path):
                # 将文件从临时目录移动到目标位置
                shutil.move(downloaded_path, local_path)
                logger.info(f"Successfully downloaded to {local_path}")
                
                # 清理临时目录
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
                    
                return True
            else:
                logger.warning(f"Downloaded path not found: {downloaded_path}")
                return False
                
        except Exception as e:
            logger.error(f"Error downloading with huggingface-cli: {str(e)}")
            # 清理临时目录
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            return False

    def download_model(self, model_type, author, max_retries=3):
        """
        统一的模型下载函数，处理生成器和鉴别器模型
        
        参数:
        - model_type: 模型类型，'generator' 或 'discriminator'
        - author: 作者名称
        - max_retries: 最大重试次数
        
        返回:
        - success: 下载是否成功
        """
        # 确定远程和本地路径
        if model_type == 'generator':
            remote_path = f"generators/{author}"
            local_path = os.path.join(self.generators_path, author)
        elif model_type == 'discriminator':
            remote_path = f"discriminators/{author}"
            local_path = os.path.join(self.discriminators_path, author)
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # 检查本地模型是否已存在
        config_path = os.path.join(local_path, "config.json")
        if os.path.exists(config_path):
            logger.info(f"Local {model_type} model already exists at {local_path}")
            return True
            
        # 尝试下载模型
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading {model_type} model for {author} (attempt {attempt+1}/{max_retries})...")
                self._wait_for_rate_limit()
                
                # 使用huggingface-cli下载
                success = self._download_model_with_cli(remote_path, local_path, self.token)
                
                if success:
                    logger.info(f"Successfully downloaded {model_type} model to {local_path}")
                    return True
                else:
                    logger.warning(f"Download attempt {attempt+1} failed")
                    
                    if attempt == max_retries - 1:
                        logger.error(f"Failed to download {model_type} model after {max_retries} attempts")
                        return False
            except Exception as e:
                if "429" in str(e):  # 速率限制错误
                    wait_time = (attempt + 1) * 10  # 等待时间指数增长
                    logger.warning(f"API rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error downloading model (attempt {attempt+1}/{max_retries}): {str(e)}")
                    
                    if attempt == max_retries - 1:
                        logger.error("Failed to download model after multiple retries")
                        return False
        return False

    def _load_model(self, author, max_retries=3):
        """加载指定作者的生成器模型，带重试机制"""
        if author not in self.available_authors:
            raise ValueError(f"作者 {author} 不可用。请从以下作者中选择: {', '.join(self.available_authors)}")
            
        if author in self.loaded_models:
            return self.loaded_models[author]

        # 检查本地模型是否存在
        local_author_path = os.path.join(self.generators_path, author)
        local_model_exists = os.path.exists(local_author_path) and os.path.exists(os.path.join(local_author_path, "config.json"))
        
        # 有token且本地不存在，尝试下载
        if not local_model_exists:
            logger.info(f"Local model not found for {author}, attempting to download")
            download_success = self.download_model('generator', author)
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

    def _load_discriminator(self, author, max_retries=3):
        """加载指定作者的鉴别器模型，带重试机制"""
        # 检查本地鉴别器模型是否存在
        local_disc_path = os.path.join(self.discriminators_path, author)
        local_best_model_path = os.path.join(local_disc_path, "best_model")  # 添加 best_model 子目录路径
        
        # 检查两种可能的路径
        local_disc_exists = os.path.exists(local_disc_path) and os.path.exists(os.path.join(local_disc_path, "config.json"))
        local_best_model_exists = os.path.exists(local_best_model_path) and os.path.exists(os.path.join(local_best_model_path, "config.json"))
        
        # 确定最终使用的路径
        final_model_path = local_best_model_path if local_best_model_exists else local_disc_path
        
        # 检查作者是否在可用列表中
        if author not in self.available_authors:
            raise ValueError(f"作者 {author} 不可用。请从以下作者中选择: {', '.join(self.available_authors)}")
        
        # 如果本地不存在，尝试下载
        if not local_disc_exists and not local_best_model_exists:
            logger.info(f"Local discriminator model not found for {author}, attempting to download")
            download_success = self.download_model('discriminator', author)
            if download_success:
                logger.info(f"Successfully downloaded discriminator model for {author}")
                # 重新检查路径
                local_disc_exists = os.path.exists(local_disc_path) and os.path.exists(os.path.join(local_disc_path, "config.json"))
                local_best_model_exists = os.path.exists(local_best_model_path) and os.path.exists(os.path.join(local_best_model_path, "config.json"))
                final_model_path = local_best_model_path if local_best_model_exists else local_disc_path
            else:
                logger.warning(f"Failed to download discriminator model for {author}")
        
        for attempt in range(max_retries):
            try:
                # 优先尝试从本地加载
                if local_disc_exists or local_best_model_exists:
                    logger.info(f"Loading local discriminator model for {author} from {final_model_path}... (attempt {attempt + 1}/{max_retries})")
                    try:
                        # 从确定的路径加载模型
                        tokenizer = BertTokenizer.from_pretrained(final_model_path)
                        model = BertForSequenceClassification.from_pretrained(final_model_path)
                        
                        # 加载标签名称
                        label_path = os.path.join(final_model_path, "label_names.json")
                        author_labels = None
                        author_idx = None
                        
                        if os.path.exists(label_path):
                            try:
                                with open(label_path, "r", encoding='utf-8') as f:
                                    author_labels = json.load(f)
                                logger.info(f"Loaded label names from {label_path}: {author_labels}")
                                
                                # 确保author在标签中
                                if author in author_labels:
                                    # 直接找到作者在标签中的索引位置
                                    author_idx = author_labels.index(author)
                                    logger.info(f"Found author {author} at index {author_idx} in label_names")
                                else:
                                    # 标签中没有当前作者，尝试查找None后的第一个位置（通常是正类别）
                                    if None in author_labels and author_labels.index(None) + 1 < len(author_labels):
                                        author_idx = author_labels.index(None) + 1
                                        logger.info(f"Author {author} not found in labels. Using index {author_idx} (after None)")
                                    else:
                                        # 假设第1个位置是正类别
                                        author_idx = 1
                                        logger.info(f"Using default index 1 for author {author}")
                            except Exception as e:
                                logger.warning(f"Error loading label names: {str(e)}")
                                # 默认使用索引1
                                author_idx = 1
                                author_labels = [None, author]
                        else:
                            logger.warning(f"Label names file not found at {label_path}")
                            # 默认使用索引1
                            author_idx = 1
                            author_labels = [None, author]
                        
                        # 创建一个元组，既包含tokenizer和model，也包含author_labels和author_idx
                        result = (tokenizer, model, author_labels, author_idx)
                        logger.info(f"Successfully loaded local discriminator model for {author}")
                        return result
                    except Exception as local_e:
                        logger.warning(f"Failed to load local discriminator: {str(local_e)}. Will try remote...")
                
                # 尝试从远程加载
                logger.info(f"Loading remote discriminator model for {author}... (attempt {attempt + 1}/{max_retries})")
                self._wait_for_rate_limit()
                
                try:
                    # 使用明确的远程路径，尝试两种可能的路径结构
                    remote_paths = [
                        f"discriminators/{author}/best_model",  # 尝试带 best_model 的路径
                        f"discriminators/{author}"              # 尝试不带 best_model 的路径
                    ]
                    
                    tokenizer = None
                    model = None
                    author_labels = None
                    author_idx = None
                    
                    for remote_path in remote_paths:
                        try:
                            logger.info(f"Attempting to load from remote path: {self.model_name}/{remote_path}")
                            
                            tokenizer = BertTokenizer.from_pretrained(
                                self.model_name,
                                subfolder=remote_path,
                                token=self.token
                            )
                            
                            model = BertForSequenceClassification.from_pretrained(
                                self.model_name,
                                subfolder=remote_path,
                                token=self.token
                            )
                            
                            # 尝试加载label_names.json
                            try:
                                from huggingface_hub import hf_hub_download
                                label_file = hf_hub_download(
                                    repo_id=self.model_name,
                                    filename=f"{remote_path}/label_names.json",
                                    token=self.token
                                )
                                
                                with open(label_file, "r", encoding='utf-8') as f:
                                    author_labels = json.load(f)
                                logger.info(f"Loaded remote label names: {author_labels}")
                                
                                # 确保author在标签中
                                if author in author_labels:
                                    # 直接找到作者的索引
                                    author_idx = author_labels.index(author)
                                    logger.info(f"Found author {author} at index {author_idx} in remote label_names")
                                else:
                                    # 标签中没有当前作者，尝试查找None后的第一个位置
                                    if None in author_labels and author_labels.index(None) + 1 < len(author_labels):
                                        author_idx = author_labels.index(None) + 1
                                        logger.info(f"Author {author} not found in remote labels. Using index {author_idx}")
                                    else:
                                        # 使用默认索引1
                                        author_idx = 1
                                        logger.info(f"Using default index 1 for author {author} (remote)")
                            except Exception as label_e:
                                logger.warning(f"Error loading remote label names: {str(label_e)}")
                                # 使用默认值
                                author_labels = [None, author]
                                author_idx = 1
                            
                            logger.info(f"Successfully loaded discriminator from {remote_path}")
                            break
                        except Exception as path_e:
                            logger.warning(f"Failed to load from {remote_path}: {str(path_e)}")
                    
                    if tokenizer is None or model is None:
                        raise ValueError("Failed to load discriminator model from any remote path")
                    
                    # 尝试保存到本地
                    try:
                        save_path = os.path.join(local_disc_path, "best_model")
                        logger.info(f"Saving discriminator model to local path: {save_path}")
                        os.makedirs(save_path, exist_ok=True)
                        tokenizer.save_pretrained(save_path)
                        model.save_pretrained(save_path)
                        
                        # 保存标签文件
                        label_save_path = os.path.join(save_path, "label_names.json")
                        with open(label_save_path, "w", encoding='utf-8') as f:
                            json.dump(author_labels, f, ensure_ascii=False, indent=2)
                        logger.info(f"Saved label names to {label_save_path}")
                        
                        logger.info(f"Successfully saved discriminator model to {save_path}")
                    except Exception as save_e:
                        logger.warning(f"Failed to save discriminator model locally: {str(save_e)}")
                    
                    # 返回四元组
                    return tokenizer, model, author_labels, author_idx
                except Exception as e:
                    logger.error(f"Error loading from remote repository: {str(e)}")
                    if "429" in str(e):  # 速率限制错误
                        wait_time = (attempt + 1) * 10  # 等待时间指数增长
                        logger.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                        time.sleep(wait_time)
                    elif attempt == max_retries - 1:
                        raise  # 最后一次尝试失败，抛出异常
            except Exception as e:
                if "429" in str(e):  # 速率限制错误
                    wait_time = (attempt + 1) * 10  # 等待时间指数增长
                    logger.warning(f"Rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Error loading discriminator model for {author}: {str(e)}")
                    if attempt == max_retries - 1:
                        raise  # 所有重试都失败，抛出异常
        
        raise RuntimeError(f"无法加载{author}的鉴别器模型，请检查网络连接和token权限")

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

    def compare_style(self, author, text1, text2, model_names=None):
        """
        比较两段文本与指定作者风格的匹配度
        
        参数:
        - author: 作者名称
        - text1: 第一段文本
        - text2: 第二段文本
        - model_names: 模型名称列表，默认为None（将自动生成）
        
        返回:
        - result: 包含比较结果的字典
        """
        try:
            # 加载鉴别器模型
            tokenizer, model, author_labels, author_idx = self._load_discriminator(author)
            
            # 评估第一段文本
            inputs1 = tokenizer(
                text1,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs1 = model(**inputs1)
                logits1 = outputs1.logits
                probs1 = F.softmax(logits1, dim=1)
                score1 = probs1[0][author_idx].item()
            
            # 评估第二段文本
            inputs2 = tokenizer(
                text2,
                return_tensors="pt",
                padding='max_length',
                truncation=True,
                max_length=512
            )
            
            with torch.no_grad():
                outputs2 = model(**inputs2)
                logits2 = outputs2.logits
                probs2 = F.softmax(logits2, dim=1)
                score2 = probs2[0][author_idx].item()
            
            # 生成模型名称（如果未提供）
            if model_names is None:
                model_names = ["Model 1", "Model 2"]
            
            # 返回比较结果
            return {
                f"{model_names[0]}_score": score1,
                f"{model_names[1]}_score": score2,
                "better_match": model_names[0] if score1 > score2 else model_names[1] if score2 > score1 else "tie",
                "author": author,
                "label_names": author_labels
            }
        except Exception as e:
            logger.error(f"Error comparing text styles: {str(e)}")
            raise
    
    def get_available_authors(self):
        """
        获取所有可用的作者列表
        
        返回:
        - authors: 可用作者的列表
        """
        return self.available_authors

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

# 简化的比较函数
def compare_style(author, local_text, deepseek_text, token=None):
    """
    比较本地模型和DeepSeek模型生成的文本与指定作者风格的匹配度
    
    参数:
    - author: 作者名称
    - local_text: 本地模型生成的文本
    - deepseek_text: DeepSeek生成的文本
    - token: Hugging Face API token，默认为None
    
    返回:
    - result: 包含比较结果的字典
    """
    api = AuthorStyleAPI(token=token)
    return api.compare_style(
        author, 
        local_text, 
        deepseek_text, 
        model_names=["local", "deepseek"]
    )