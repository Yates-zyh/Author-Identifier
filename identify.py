import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Dict, List
import logging
import dotenv
import time
from datetime import datetime
import shutil
import subprocess

# 加载环境变量
dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("author_identifier")

class AuthorIdentifier:
    """
    Author Identifier
    Used to predict the style of text authors
    """
    
    def __init__(self, model_path: str = None, device: str = None, token: str = None):
        """
        Initialize the Author Identifier
        
        Parameters:
        - model_path: Path to the model, default is None, will use "author_style_model"
        - device: Device to use, default is None (auto-detection)
        - token: Hugging Face API token, default is None (will use IDENTIFICATION_TOKEN if present)
        """
        # 模型路径设置
        self.default_local_path = "author_style_model"
        self.model_path = model_path or self.default_local_path
        self.remote_model_path = "Yates-zyh/author_identifier"
        
        # 设备设置
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Token设置
        self.token = token or os.environ.get("IDENTIFICATION_TOKEN")
        
        # 速率限制变量
        self.last_request_time = None
        self.min_request_interval = 2  # 最小请求间隔（秒）
        
        # 创建本地模型目录（如果不存在）
        if not os.path.exists(self.model_path):
            logger.info(f"Creating local model directory: {self.model_path}")
            os.makedirs(self.model_path, exist_ok=True)
        
        # 检查本地模型是否存在
        self.local_model_exists = os.path.exists(os.path.join(self.model_path, "config.json"))
        
        # 加载模型
        self._initialize_model()
        
        # 加载模型元数据
        self.metadata = self._load_metadata()
    
    def _initialize_model(self):
        """初始化模型：如果有token则下载模型到本地并加载，否则直接从远程加载"""
        if self.token:
            # 有token，尝试下载模型到本地
            logger.info("Token available, attempting to download model to local directory")
            download_success = self._download_model_to_local()
            
            if download_success:
                # 下载成功，从本地加载
                logger.info("Model downloaded successfully, loading from local path")
                self._load_local_model()
            else:
                # 下载失败，直接从远程加载
                logger.warning("Failed to download model, loading directly from remote")
                self._load_remote_model()
        else:
            # 没有token
            if self.local_model_exists:
                # 本地模型存在，直接从本地加载
                logger.info("No token provided but local model exists, loading from local path")
                self._load_local_model()
            else:
                # 本地模型不存在，直接从远程加载
                logger.info("No token provided and no local model, loading directly from remote")
                self._load_remote_model()
    
    def _wait_for_rate_limit(self):
        """确保请求间隔符合速率限制"""
        if self.last_request_time is not None:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = datetime.now()
    
    def _download_model_to_local(self, max_retries=3):
        """下载模型到本地目录"""
        if not self.token:
            logger.error("No Hugging Face token available for model download")
            # 提示用户输入token
            print("需要Hugging Face令牌(token)才能下载模型")
            user_token = input("请输入您的Hugging Face令牌(token)，或直接按Enter跳过: ")
            if user_token:
                self.token = user_token
                logger.info("User provided token will be used for download")
            else:
                logger.warning("No token provided, will try to use from-pretrained")
                return False
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading model from {self.remote_model_path} to {self.model_path} (attempt {attempt+1}/{max_retries})...")
                self._wait_for_rate_limit()
                
                # 创建临时目录
                temp_dir = f"{self.model_path}_temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                # 使用huggingface-cli下载整个模型仓库
                cmd = [
                    "huggingface-cli", "download",
                    "--token", self.token,
                    "--local-dir", temp_dir,
                    self.remote_model_path
                ]
                
                # 执行命令
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # 检查命令是否成功
                if result.returncode != 0:
                    logger.warning(f"huggingface-cli download failed: {result.stderr}")
                    raise Exception(f"huggingface-cli download failed: {result.stderr}")
                
                # 成功下载后，移动文件到最终目录
                if os.path.exists(self.model_path):
                    shutil.rmtree(self.model_path)
                shutil.move(temp_dir, self.model_path)
                
                logger.info(f"Model successfully downloaded to {self.model_path}")
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
    
    def _load_local_model(self):
        """从本地路径加载模型"""
        try:
            logger.info(f"Loading model from local path: {self.model_path}")
            self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
            self.model = BertForSequenceClassification.from_pretrained(self.model_path)
            
            # 加载本地标签
            try:
                with open(os.path.join(self.model_path, "label_names.json"), 'r', encoding='utf-8') as f:
                    self.label_names = json.load(f)
            except FileNotFoundError:
                logger.warning(f"Label file not found: {os.path.join(self.model_path, 'label_names.json')}")
                self.label_names = [f"Category{i}" for i in range(self.model.config.num_labels)]
            
            # 将模型移到指定设备
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded {len(self.label_names)} labels: {', '.join(self.label_names)}")
            return True
        except Exception as e:
            logger.error(f"Error loading local model: {str(e)}")
            return False
    
    def _load_remote_model(self):
        """直接从远程仓库加载模型"""
        try:
            logger.info(f"Loading model directly from remote: {self.remote_model_path}")
            # 直接从远程加载模型，无论是否有token
            kwargs = {"token": self.token} if self.token else {}
            
            # 加载tokenizer和模型
            self.tokenizer = BertTokenizer.from_pretrained(
                self.remote_model_path,
                **kwargs
            )
            
            self.model = BertForSequenceClassification.from_pretrained(
                self.remote_model_path,
                **kwargs
            )
            
            # 尝试加载标签信息
            try:
                # 创建临时目录下载标签文件
                temp_dir = f"{self.model_path}_temp"
                os.makedirs(temp_dir, exist_ok=True)
                
                # 准备下载命令
                cmd = ["huggingface-cli", "download"]
                if self.token:
                    cmd.extend(["--token", self.token])
                cmd.extend([
                    "--include", "label_names.json",
                    "--local-dir", temp_dir,
                    self.remote_model_path
                ])
                
                # 执行命令
                result = subprocess.run(cmd, capture_output=True, text=True)
                
                # 检查命令是否成功
                if result.returncode != 0:
                    raise Exception(f"Failed to download label file: {result.stderr}")
                
                # 读取下载的标签文件
                label_file = os.path.join(temp_dir, "label_names.json")
                if os.path.exists(label_file):
                    with open(label_file, 'r', encoding='utf-8') as f:
                        self.label_names = json.load(f)
                else:
                    raise FileNotFoundError("Label file not found after download")
                
            except Exception as e:
                logger.warning(f"Error loading remote labels: {str(e)}")
                # 使用模型的默认标签
                self.label_names = [f"Category{i}" for i in range(self.model.config.num_labels)]
            finally:
                # 清理临时目录
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)
            
            # 将模型移到指定设备
            self.model.to(self.device)
            self.model.eval()
            
            logger.info(f"Loaded {len(self.label_names)} labels: {', '.join(self.label_names)}")
            return True
        except Exception as e:
            logger.error(f"Error loading remote model: {str(e)}")
            return False
    
    def _load_metadata(self) -> Dict:
        """加载模型元数据"""
        try:
            # 先尝试从本地加载
            if os.path.exists(os.path.join(self.model_path, "model_metadata.json")):
                try:
                    with open(os.path.join(self.model_path, "model_metadata.json"), 'r', encoding='utf-8') as f:
                        metadata = json.load(f)
                    logger.info(f"Local model metadata loaded")
                    return metadata
                except Exception as e:
                    logger.warning(f"Failed to load local model metadata: {str(e)}")
            
            # 如果本地没有，且有token，尝试从远程加载
            if self.token:
                try:
                    self._wait_for_rate_limit()
                    
                    # 创建临时目录下载元数据文件
                    temp_dir = f"{self.model_path}_temp_metadata"
                    os.makedirs(temp_dir, exist_ok=True)
                    
                    # 使用huggingface-cli下载元数据文件
                    cmd = [
                        "huggingface-cli", "download",
                        "--token", self.token,
                        "--include", "model_metadata.json",
                        "--local-dir", temp_dir,
                        self.remote_model_path
                    ]
                    
                    # 执行命令
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    
                    # 检查命令是否成功
                    if result.returncode != 0:
                        raise Exception(f"Failed to download metadata file: {result.stderr}")
                    
                    # 读取下载的元数据文件
                    metadata_file = os.path.join(temp_dir, "model_metadata.json")
                    if os.path.exists(metadata_file):
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        # 清理临时目录
                        shutil.rmtree(temp_dir)
                        logger.info(f"Remote model metadata loaded")
                        return metadata
                    else:
                        raise FileNotFoundError("Metadata file not found after download")
                    
                except Exception as e:
                    logger.warning(f"Failed to load remote model metadata: {str(e)}")
                    # 清理临时目录
                    if os.path.exists(temp_dir):
                        shutil.rmtree(temp_dir)
            
            # 无法加载任何元数据，返回空字典
            logger.warning(f"No metadata could be loaded")
            return {}
        except Exception as e:
            logger.error(f"Error in _load_metadata: {str(e)}")
            return {}
    
    def _batch_texts(self, text: str, max_length: int = 512, overlap: int = 128, 
                    min_chunk_length: int = 100) -> List[str]:
        """
        Split long text into multiple overlapping batches for processing
        
        Parameters:
        - text: Input text
        - max_length: Maximum token length
        - overlap: Number of overlapping tokens
        - min_chunk_length: Minimum chunk length (tokens)
        
        Returns:
        - chunks: List of text batches
        """
        # Ensure text is cleaned, removing extra whitespaces
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        # Use sliding window to split text
        while start_idx < len(tokens):
            # Ensure not exceeding text length
            end_idx = min(start_idx + max_length, len(tokens))
            
            window_tokens = tokens[start_idx:end_idx]
            
            # Only add chunks that are long enough
            if len(window_tokens) >= min_chunk_length:
                window_text = self.tokenizer.decode(window_tokens)
                chunks.append(window_text)
            
            # Exit loop if reaching the end of text
            if end_idx == len(tokens):
                break
                
            # Update the start position of the next window (consider overlap)
            start_idx += (max_length - overlap)
        
        return chunks
    
    def analyze_text(self, text: str, confidence_threshold: float = 0.7, 
                    return_all_chunks: bool = False) -> Dict:
        """
        Analyze the style of text authors
        
        Parameters:
        - text: Input text
        - confidence_threshold: Confidence threshold, below which "Unknown Author" will be returned
        - return_all_chunks: Whether to return analysis results for all text chunks
        
        Returns:
        - result: Dictionary containing prediction results
        """
        # If the text is too long, split into multiple chunks for analysis
        if len(text.split()) > 200:  # Approximately more than 200 words, split into chunks
            chunks = self._batch_texts(text)
            logger.info(f"Text divided into {len(chunks)} chunks for analysis")
        else:
            chunks = [text]
            
        all_chunk_results = []
        all_probabilities = []
        
        # Process each text chunk
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                
            chunk_result = self._analyze_single_chunk(chunk, confidence_threshold)
            all_chunk_results.append(chunk_result)
            all_probabilities.append(chunk_result["raw_probabilities"])
        
        # Aggregate probabilities from multiple chunks
        if len(all_probabilities) > 1:
            # Combine probabilities from all chunks (using average)
            avg_probabilities = np.mean(all_probabilities, axis=0)
            
            # Get the highest probability and its corresponding category
            max_prob_idx = np.argmax(avg_probabilities)
            max_prob = avg_probabilities[max_prob_idx]
            
            # Create final result based on aggregated probabilities
            if max_prob < confidence_threshold:
                final_author = "Unknown Author"
                final_confidence = 1 - max_prob  # Use uncertainty as confidence
            else:
                final_author = self.label_names[max_prob_idx]
                final_confidence = max_prob
                
            # Generate final result
            final_result = {
                "predicted_author": final_author,
                "confidence": float(final_confidence),
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, avg_probabilities)},
                "num_chunks_analyzed": len(chunks)
            }
            
            # Count predictions for each chunk
            author_counts = {}
            for res in all_chunk_results:
                author = res["predicted_author"]
                if author not in author_counts:
                    author_counts[author] = 0
                author_counts[author] += 1
                
            final_result["author_distribution"] = author_counts
            
            # If needed, add results for each chunk
            if return_all_chunks:
                final_result["chunk_results"] = all_chunk_results
        else:
            # Only one chunk, directly use its result
            final_result = all_chunk_results[0]
            final_result["num_chunks_analyzed"] = 1
            
        return final_result
    
    def _analyze_single_chunk(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Analyze the style of a single text chunk
        Parameters:
        - text: Input text
        - confidence_threshold: Confidence threshold
        Returns:
        - result: Dictionary containing prediction results
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the highest probability and its corresponding category
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()
        predicted_class = predicted_class.item()
        
        # Save raw probabilities for further processing
        raw_probabilities = probabilities[0].cpu().numpy()
        
        if max_prob < confidence_threshold:
            result = {
                "predicted_author": "Unknown Author",
                "confidence": 1 - max_prob,  # Use uncertainty as confidence
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, raw_probabilities)},
                "raw_probabilities": raw_probabilities
            }
        else:
            result = {
                "predicted_author": self.label_names[predicted_class],
                "confidence": max_prob,
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, raw_probabilities)},
                "raw_probabilities": raw_probabilities
            }
        
        return result
    
    def analyze_file(self, file_path: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Analyze the style of text authors in a file
        Parameters:
        - file_path: Path to the file
        - confidence_threshold: Confidence threshold
        Returns:
        - result: Dictionary containing prediction results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = self.analyze_text(text, confidence_threshold)
            result["file_path"] = file_path
            return result
        except Exception as e:
            logger.error(f"Failed to analyze file: {str(e)}")
            return {
                "error": f"Failed to analyze file: {str(e)}",
                "file_path": file_path
            }
    
    def get_model_info(self) -> Dict:
        info = {
            "model_path": self.model_path if self.local_model_exists else self.remote_model_path,
            "mode": "local" if self.local_model_exists else "remote",
            "device": str(self.device),
            "labels": self.label_names,
            "num_labels": len(self.label_names),
        }
        
        # Add metadata information
        info.update(self.metadata)
        
        return info

# 简化的API函数
def analyze_text(text: str, confidence_threshold: float = 0.7, token: str = None) -> Dict:
    identifier = AuthorIdentifier(token=token)
    return identifier.analyze_text(text, confidence_threshold)