import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("author_identifier")

class AuthorIdentifier:
    """
    作者身份识别器
    用于预测文本的作者风格
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        初始化作者识别器
        
        参数:
        - model_path: 模型路径，默认为None，会尝试自动查找最新模型
        - device: 使用的设备，默认为None（自动检测）
        """
        self.model_path = "../author_style_model"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"加载模型: {self.model_path}")
        logger.info(f"使用设备: {self.device}")
        
        # 加载分词器和模型
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # 加载标签
        try:
            with open(os.path.join(self.model_path, "label_names.json"), 'r', encoding='utf-8') as f:
                self.label_names = json.load(f)
            logger.info(f"已加载{len(self.label_names)}个标签: {', '.join(self.label_names)}")
        except FileNotFoundError:
            logger.warning(f"标签文件不存在: {os.path.join(self.model_path, 'label_names.json')}")
            self.label_names = [f"类别{i}" for i in range(self.model.config.num_labels)]
            
        # 加载模型元数据
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """加载模型元数据"""
        metadata_path = os.path.join(self.model_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"已加载模型元数据")
                return metadata
            except Exception as e:
                logger.warning(f"加载模型元数据失败: {str(e)}")
        
        logger.warning(f"元数据文件不存在: {metadata_path}")
        return {}
    
    def _batch_texts(self, text: str, max_length: int = 512, overlap: int = 128, 
                     min_chunk_length: int = 100) -> List[str]:
        """
        将长文本分成多个重叠的批次进行处理
        
        参数:
        - text: 输入文本
        - max_length: 最大token长度
        - overlap: 重叠token数
        - min_chunk_length: 最小chunk长度（tokens）
        
        返回:
        - chunks: 文本批次列表
        """
        # 确保文本进行了清理，移除多余空白符
        text = ' '.join(text.split())
        
        # 分词
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        # 使用滑动窗口切分文本
        while start_idx < len(tokens):
            # 确保不超出文本长度
            end_idx = min(start_idx + max_length, len(tokens))
            
            window_tokens = tokens[start_idx:end_idx]
            
            # 只添加足够长的chunk
            if len(window_tokens) >= min_chunk_length:
                window_text = self.tokenizer.decode(window_tokens)
                chunks.append(window_text)
            
            # 如果已经到达文本末尾，退出循环
            if end_idx == len(tokens):
                break
                
            # 更新下一个窗口的起始位置（考虑重叠）
            start_idx += (max_length - overlap)
        
        return chunks
    
    def analyze_text(self, text: str, confidence_threshold: float = 0.7, 
                     return_all_chunks: bool = False) -> Dict:
        """
        分析文本的作者风格
        
        参数:
        - text: 输入文本
        - confidence_threshold: 置信度阈值，低于此值将返回"未知作家"
        - return_all_chunks: 是否返回所有文本块的分析结果
        
        返回:
        - result: 包含预测结果的字典
        """
        # 如果文本太长，分成多个块进行分析
        if len(text.split()) > 200:  # 大约超过200个单词就分块
            chunks = self._batch_texts(text)
            logger.info(f"文本被分为{len(chunks)}个块进行分析")
        else:
            chunks = [text]
            
        all_chunk_results = []
        all_probabilities = []
        
        # 处理每个文本块
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(f"处理块 {i+1}/{len(chunks)}...")
                
            chunk_result = self._analyze_single_chunk(chunk, confidence_threshold)
            all_chunk_results.append(chunk_result)
            all_probabilities.append(chunk_result["raw_probabilities"])
        
        # 汇总多个块的概率
        if len(all_probabilities) > 1:
            # 合并所有块的概率（使用平均值）
            avg_probabilities = np.mean(all_probabilities, axis=0)
            
            # 获取最高概率及其对应的类别
            max_prob_idx = np.argmax(avg_probabilities)
            max_prob = avg_probabilities[max_prob_idx]
            
            # 基于汇总概率制作最终结果
            if max_prob < confidence_threshold:
                final_author = "未知作家"
                final_confidence = 1 - max_prob  # 使用不确定性作为置信度
            else:
                final_author = self.label_names[max_prob_idx]
                final_confidence = max_prob
                
            # 生成最终结果
            final_result = {
                "predicted_author": final_author,
                "confidence": float(final_confidence),
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, avg_probabilities)},
                "num_chunks_analyzed": len(chunks)
            }
            
            # 统计各个块的预测
            author_counts = {}
            for res in all_chunk_results:
                author = res["predicted_author"]
                if author not in author_counts:
                    author_counts[author] = 0
                author_counts[author] += 1
                
            final_result["author_distribution"] = author_counts
            
            # 如果需要，添加每个块的结果
            if return_all_chunks:
                final_result["chunk_results"] = all_chunk_results
        else:
            # 只有一个块，直接使用其结果
            final_result = all_chunk_results[0]
            final_result["num_chunks_analyzed"] = 1
            
        return final_result
    
    def _analyze_single_chunk(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        分析单个文本块的作者风格
        
        参数:
        - text: 输入文本
        - confidence_threshold: 置信度阈值
        
        返回:
        - result: 包含预测结果的字典
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
        
        # 获取最高概率及其对应的类别
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()
        predicted_class = predicted_class.item()
        
        # 保存原始概率以便后续处理
        raw_probabilities = probabilities[0].cpu().numpy()
        
        if max_prob < confidence_threshold:
            result = {
                "predicted_author": "未知作家",
                "confidence": 1 - max_prob,  # 使用不确定性作为置信度
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
        分析文件中的文本作者风格
        
        参数:
        - file_path: 文件路径
        - confidence_threshold: 置信度阈值
        
        返回:
        - result: 包含预测结果的字典
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = self.analyze_text(text, confidence_threshold)
            result["file_path"] = file_path
            return result
        except Exception as e:
            logger.error(f"分析文件失败: {str(e)}")
            return {
                "error": f"分析文件失败: {str(e)}",
                "file_path": file_path
            }
    
    def get_model_info(self) -> Dict:
        """
        获取模型信息
        
        返回:
        - info: 包含模型信息的字典
        """
        info = {
            "model_path": self.model_path,
            "device": str(self.device),
            "labels": self.label_names,
            "num_labels": len(self.label_names),
        }
        
        # 添加元数据信息
        info.update(self.metadata)
        
        return info

# 简化的API函数
def analyze_text(text: str, confidence_threshold: float = 0.7) -> Dict:
    """
    简化的API函数，用于分析文本作者风格
    
    参数:
    - text: 输入文本
    - confidence_threshold: 置信度阈值
    
    返回:
    - result: 包含预测结果的字典
    """
    identifier = AuthorIdentifier()
    return identifier.analyze_text(text, confidence_threshold)