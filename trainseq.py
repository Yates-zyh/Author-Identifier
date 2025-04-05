import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler, TensorDataset
from transformers import (
    GPT2LMHeadModel, 
    GPT2Tokenizer, 
    GPT2Config,
    BertForSequenceClassification, 
    BertTokenizer,
    get_linear_schedule_with_warmup
)
from torch.optim import AdamW
import numpy as np
import json
import os
import logging
import nltk
from nltk.tokenize import sent_tokenize
import random
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import re
import shutil
from collections import defaultdict

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

class AuthorDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.encodings = tokenizer(texts, truncation=True, padding='max_length', 
                                  max_length=max_length, return_tensors='pt')
        self.labels = self.encodings.input_ids.clone()
    
    def __len__(self):
        return len(self.encodings.input_ids)
    
    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

class GPT2StyleGenerator:
    def __init__(self, author, data_dir="data", output_dir="gpt2_generator", 
                model_name="gpt2", max_length=256, batch_size=4):
        """
        初始化GPT-2风格生成器
        
        参数:
        - author: 目标作者名称
        - data_dir: 包含作者数据的目录
        - output_dir: 保存生成文本和模型的目录
        - model_name: GPT-2模型名称
        - max_length: 最大序列长度
        - batch_size: 批处理大小
        """
        self.author = author
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, author)
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载tokenizer和模型
        logger.info(f"Loading GPT-2 tokenizer and model for {author}...")
        self.tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # 使用默认配置初始化模型
        self.model = GPT2LMHeadModel.from_pretrained(model_name)
        self.model.to(device)
        
        # 加载作者文本
        self.texts = self.load_author_texts()
        logger.info(f"Loaded {len(self.texts)} text samples for author {author}")
        
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        
    def load_author_texts(self, chunk_size=256, overlap=50):
        """加载作者文本并分块"""
        texts = []
        author_dir = os.path.join(self.data_dir, self.author)
        
        if not os.path.exists(author_dir):
            logger.error(f"Author directory {author_dir} not found")
            return texts
        
        # 逐个文件处理，而不是先合并所有文本
        for filename in os.listdir(author_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(author_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 分句处理
                    sentences = sent_tokenize(content)
                    
                    # 按句子创建文本块
                    current_chunk = ""
                    current_tokens = 0
                    
                    for sentence in sentences:
                        # 估计句子的token数量，避免直接编码
                        sentence_tokens = len(sentence.split()) * 1.3  # 估计值，单词数*1.3作为token数的粗略估计
                        
                        if current_tokens + sentence_tokens > chunk_size:
                            if current_chunk and len(current_chunk.split()) > 20:  # 确保块足够长
                                texts.append(current_chunk)
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            current_chunk += " " + sentence
                            current_tokens += sentence_tokens
                    
                    # 添加最后一个块
                    if current_chunk and len(current_chunk.split()) > 20:
                        texts.append(current_chunk)
                        
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
        
        logger.info(f"Loaded {len(texts)} text samples for author {self.author}")
        return texts
    
    def prepare_data(self):
        """准备数据集"""
        # 创建数据集
        dataset = AuthorDataset(self.texts, self.tokenizer, max_length=self.max_length)
        
        # 分割为训练集和验证集
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        # 创建数据加载器
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def train(self, epochs=3, learning_rate=5e-5, warmup_steps=100):
        """训练GPT-2模型"""
        logger.info(f"Starting training for author {self.author}...")
        
        # 准备数据
        train_dataloader, val_dataloader = self.prepare_data()
        
        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=len(train_dataloader) * epochs
        )
        
        # 训练循环
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            # 训练
            self.model.train()
            train_loss = 0
            train_samples = 0
            
            for batch in tqdm(train_dataloader, desc=f"Training epoch {epoch+1}/{epochs}"):
                # 将数据移至设备
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)
                
                # 前向传播
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                train_loss += loss.item() * input_ids.size(0)
                train_samples += input_ids.size(0)
                
                # 反向传播和优化
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            
            train_loss = train_loss / train_samples
            
            # 验证
            self.model.eval()
            val_loss = 0
            val_samples = 0
            
            with torch.no_grad():
                for batch in tqdm(val_dataloader, desc=f"Validating epoch {epoch+1}/{epochs}"):
                    # 将数据移至设备
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    labels = batch['labels'].to(device)
                    
                    # 前向传播
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    val_loss += loss.item() * input_ids.size(0)
                    val_samples += input_ids.size(0)
            
            val_loss = val_loss / val_samples
            
            logger.info(f"Epoch {epoch+1}/{epochs}, Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}")
            
            # 保存最佳模型
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model("best_model")
                logger.info(f"Saved best model with val loss: {val_loss:.4f}")
            
            # 生成样本文本
            self.generate_samples(num_samples=5, file_prefix=f"epoch_{epoch+1}")
        
        # 训练完成后保存最终模型
        self.save_model("final_model")
        logger.info(f"Training completed for author {self.author}")
    
    def save_model(self, model_name):
        """保存模型"""
        output_path = os.path.join(self.output_dir, model_name)
        os.makedirs(output_path, exist_ok=True)
        
        self.model.save_pretrained(output_path)
        self.tokenizer.save_pretrained(output_path)
        logger.info(f"Model saved to {output_path}")
    
    def load_model(self, model_name="best_model"):
        """加载保存的模型"""
        model_path = os.path.join(self.output_dir, model_name)
        
        if os.path.exists(model_path):
            self.model = GPT2LMHeadModel.from_pretrained(model_path)
            self.model.to(device)
            self.tokenizer = GPT2Tokenizer.from_pretrained(model_path)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model {model_path} not found, using base model")
    
    def generate_samples(self, num_samples=10, max_length=200, temperature=1.0, top_k=50, 
                         top_p=0.95, file_prefix="sample"):
        """生成样本文本"""
        logger.info(f"Generating {num_samples} samples...")
        
        self.model.eval()
        samples = []
        
        for i in range(num_samples):
            # 生成一个样本
            input_text = random.choice([
                "The", "She", "He", "It", "There", "In", "When", "A", "My", "Once"
            ])
            inputs = self.tokenizer(input_text, return_tensors="pt", padding=True)
            input_ids = inputs["input_ids"].to(device)
            attention_mask = inputs["attention_mask"].to(device)
            
            # 生成文本
            output = self.model.generate(
                input_ids,
                attention_mask=attention_mask,  # 明确传递attention_mask
                max_length=max_length,
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=1.2,
                do_sample=True,
                num_return_sequences=1,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
            samples.append(generated_text)
            
            # 如果是训练中的样本，打印查看进度
            if i == 0:
                logger.info(f"Sample: {generated_text[:100]}...")
        
        # 保存样本
        output_path = os.path.join(self.output_dir, f"{file_prefix}_samples.txt")
        with open(output_path, "w", encoding="utf-8") as f:
            for i, sample in enumerate(samples):
                f.write(f"Sample {i+1}:\n")
                f.write(sample)
                f.write("\n\n" + "-"*50 + "\n\n")
        
        logger.info(f"Samples saved to {output_path}")
        return samples

class BERTDiscriminator:
    """BERT判别器类"""
    def __init__(self, authors, data_dir="data", output_dir="bert_discriminator", 
                model_name="bert-base-uncased", max_length=256, batch_size=8):
        self.authors = authors
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 加载tokenizer
        logger.info("Loading BERT tokenizer and model...")
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        
        # 初始化模型
        self.model = BertForSequenceClassification.from_pretrained(
            model_name,
            num_labels=len(authors) + 1  # +1 for "Unknown" author
        )
        self.model.to(device)
        
        # 创建作者索引映射
        self.author_indices = {author: idx for idx, author in enumerate(authors)}
        self.author_indices["Unknown"] = len(authors)  # 添加"未知"作者
        
        # 设置随机种子
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
    
    def prepare_data(self, max_samples_per_author=None, nltk_samples=None):
        """准备数据集"""
        all_texts = []
        all_labels = []
        
        # 为每个作者加载文本
        for author in self.authors:
            texts = self.load_author_texts(author)
            
            # 如果需要限制样本数量
            if max_samples_per_author and len(texts) > max_samples_per_author:
                texts = random.sample(texts, max_samples_per_author)
            
            all_texts.extend(texts)
            all_labels.extend([self.author_indices[author]] * len(texts))
            logger.info(f"Author {author}: {len(texts)} samples")
        
        # 添加"未知"作者样本（可选）
        if nltk_samples:
            unknown_texts = self.load_nltk_samples(nltk_samples)
            all_texts.extend(unknown_texts)
            all_labels.extend([self.author_indices["Unknown"]] * len(unknown_texts))
            logger.info(f"Unknown author: {len(unknown_texts)} samples")
        
        # 编码文本
        encodings = self.tokenizer(
            all_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # 创建数据集
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']
        labels = torch.tensor(all_labels)
        
        # 分割为训练集和验证集
        train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
            input_ids, attention_masks, labels, test_size=0.15, random_state=42, stratify=labels
        )
        
        # 创建数据加载器
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_dataloader = DataLoader(
            train_data,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        val_data = TensorDataset(val_inputs, val_masks, val_labels)
        val_dataloader = DataLoader(
            val_data,
            batch_size=self.batch_size,
            shuffle=False
        )
        
        return train_dataloader, val_dataloader
    
    def load_author_texts(self, author, chunk_size=256, overlap=50):
        """加载作者文本并分块"""
        texts = []
        author_dir = os.path.join(self.data_dir, author)
        
        if not os.path.exists(author_dir):
            logger.error(f"Author directory {author_dir} not found")
            return texts
        
        # 逐个文件处理，而不是先合并所有文本
        for filename in os.listdir(author_dir):
            if filename.endswith('.txt'):
                filepath = os.path.join(author_dir, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()
                    
                    # 分句处理
                    sentences = sent_tokenize(content)
                    
                    # 按句子创建文本块
                    current_chunk = ""
                    current_tokens = 0
                    
                    for sentence in sentences:
                        # 估计句子的token数量，避免直接编码
                        sentence_tokens = len(sentence.split()) * 1.3  # 估计值，单词数*1.3作为token数的粗略估计
                        
                        if current_tokens + sentence_tokens > chunk_size:
                            if current_chunk and len(current_chunk.split()) > 20:  # 确保块足够长
                                texts.append(current_chunk)
                            current_chunk = sentence
                            current_tokens = sentence_tokens
                        else:
                            current_chunk += " " + sentence
                            current_tokens += sentence_tokens
                    
                    # 添加最后一个块
                    if current_chunk and len(current_chunk.split()) > 20:
                        texts.append(current_chunk)
                        
                except Exception as e:
                    logger.error(f"Error reading {filepath}: {e}")
        
        logger.info(f"Loaded {len(texts)} text samples for author {author}")
        return texts
    
    def load_nltk_samples(self, num_samples):
        """加载NLTK语料库样本作为"未知"作者样本"""
        try:
            # 确保NLTK语料库已下载
            nltk.download('gutenberg')
            nltk.download('brown')
            nltk.download('reuters')
            
            samples = []
            
            # 从Gutenberg语料库加载样本
            for fileid in nltk.corpus.gutenberg.fileids()[:10]:
                text = nltk.corpus.gutenberg.raw(fileid)
                samples.extend(self.chunk_text(text))
            
            # 从Brown语料库加载样本
            for fileid in nltk.corpus.brown.fileids()[:20]:
                text = nltk.corpus.brown.raw(fileid)
                samples.extend(self.chunk_text(text))
            
            # 从Reuters语料库加载样本
            for fileid in nltk.corpus.reuters.fileids()[:20]:
                text = nltk.corpus.reuters.raw(fileid)
                samples.extend(self.chunk_text(text))
            
            # 限制样本数量
            if len(samples) > num_samples:
                samples = random.sample(samples, num_samples)
            
            return samples
            
        except Exception as e:
            logger.error(f"Error loading NLTK samples: {e}")
            return []
    
    def chunk_text(self, text):
        """将文本分块，避免直接编码长文本"""
        chunks = []
        sentences = sent_tokenize(text)
        
        current_chunk = ""
        current_tokens = 0
        chunk_size = 256
        
        for sentence in sentences:
            # 改用words估计token数量，避免直接编码
            sentence_tokens = len(sentence.split()) * 1.3  # 粗略估计
            
            if current_tokens + sentence_tokens > chunk_size:
                if current_chunk and len(current_chunk.split()) > 20:
                    chunks.append(current_chunk)
                current_chunk = sentence
                current_tokens = sentence_tokens
            else:
                current_chunk += " " + sentence
                current_tokens += sentence_tokens
        
        # 添加最后一个块
        if current_chunk and len(current_chunk.split()) > 20:
            chunks.append(current_chunk)
        
        return chunks
    
    def train(self, epochs=3, learning_rate=2e-5):
        """训练BERT判别器"""
        logger.info("Starting BERT discriminator training...")
        
        # 准备数据
        train_dataloader, val_dataloader = self.prepare_data(
            max_samples_per_author=1000,
            nltk_samples=1000
        )
        
        # 设置优化器
        optimizer = AdamW(self.model.parameters(), lr=learning_rate, eps=1e-8)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(len(train_dataloader) * 0.1),
            num_training_steps=len(train_dataloader) * epochs
        )
        
        # 反向作者索引映射
        idx_to_author = {idx: author for author, idx in self.author_indices.items()}
        
        # 训练循环
        best_accuracy = 0
        
        for epoch in range(epochs):
            logger.info(f"======== Epoch {epoch + 1} / {epochs} ========")
            
            # 训练
            self.model.train()
            total_train_loss = 0
            
            for batch in tqdm(train_dataloader, desc="Training"):
                optimizer.zero_grad()
                
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_train_loss += loss.item()
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
            
            avg_train_loss = total_train_loss / len(train_dataloader)
            logger.info(f"Average training loss: {avg_train_loss}")
            
            # 验证
            self.model.eval()
            total_eval_accuracy = 0
            total_eval_loss = 0
            all_preds = []
            all_labels = []
            
            for batch in tqdm(val_dataloader, desc="Validation"):
                with torch.no_grad():
                    input_ids = batch[0].to(device)
                    attention_mask = batch[1].to(device)
                    labels = batch[2].to(device)
                    
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    
                    loss = outputs.loss
                    total_eval_loss += loss.item()
                    
                    logits = outputs.logits
                    predictions = torch.argmax(logits, dim=1)
                    
                    all_preds.extend(predictions.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy())
                    
                    accuracy = (predictions == labels).float().mean().item()
                    total_eval_accuracy += accuracy
            
            avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
            avg_val_loss = total_eval_loss / len(val_dataloader)
            
            # 计算分类报告
            from sklearn.metrics import classification_report, confusion_matrix
            report = classification_report(
                all_labels, 
                all_preds, 
                target_names=[idx_to_author[i] for i in range(len(idx_to_author))],
                digits=4
            )
            
            logger.info(f"Validation Accuracy: {avg_val_accuracy:.4f}")
            logger.info(f"Validation Loss: {avg_val_loss:.4f}")
            logger.info(f"Classification Report:\n{report}")
            
            # 保存最佳模型
            if avg_val_accuracy > best_accuracy:
                best_accuracy = avg_val_accuracy
                self.save_model()
                logger.info(f"Saved best model with accuracy: {best_accuracy:.4f}")
        
        logger.info("BERT discriminator training completed!")
    
    def save_model(self):
        """保存模型"""
        model_path = os.path.join(self.output_dir, "best_model")
        os.makedirs(model_path, exist_ok=True)
        
        self.model.save_pretrained(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # 保存作者标签
        label_names = [None] * (len(self.author_indices))
        for author, idx in self.author_indices.items():
            label_names[idx] = author
        
        with open(os.path.join(model_path, "label_names.json"), 'w') as f:
            json.dump(label_names, f)
        
        logger.info(f"Model saved to {model_path}")
    
    def load_model(self):
        """加载保存的模型"""
        model_path = os.path.join(self.output_dir, "best_model")
        
        if os.path.exists(model_path):
            self.model = BertForSequenceClassification.from_pretrained(model_path)
            self.model.to(device)
            self.tokenizer = BertTokenizer.from_pretrained(model_path)
            
            # 加载作者标签
            with open(os.path.join(model_path, "label_names.json"), 'r') as f:
                label_names = json.load(f)
            
            self.author_indices = {author: idx for idx, author in enumerate(label_names)}
            
            logger.info(f"Loaded model from {model_path}")
        else:
            logger.warning(f"Model {model_path} not found")
    
    def evaluate_text(self, text):
        """评估文本并返回作者概率，处理长文本"""
        self.model.eval()
        
        # 如果文本太长，分块处理
        words = text.split()
        if len(words) > 200:  # 估计超过512 tokens的文本
            # 将文本分成较小的段落
            chunks = []
            for i in range(0, len(words), 200):
                chunk = " ".join(words[i:i+200])
                chunks.append(chunk)
            
            # 处理每个块并取平均
            all_probs = {}
            for chunk in chunks:
                chunk_probs = self._evaluate_chunk(chunk)
                # 累积概率
                for author, prob in chunk_probs.items():
                    all_probs[author] = all_probs.get(author, 0) + prob
            
            # 计算平均概率
            author_probs = {author: prob/len(chunks) for author, prob in all_probs.items()}
        else:
            author_probs = self._evaluate_chunk(text)
        
        return author_probs

    def _evaluate_chunk(self, text):
        """评估单个文本块"""
        # 编码文本，确保截断
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        ).to(device)
        
        # 预测
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probs = F.softmax(logits, dim=1)
        
        # 创建作者概率字典
        author_probs = {}
        for author, idx in self.author_indices.items():
            author_probs[author] = probs[0][idx].item()
        
        return author_probs

class StyleGAN:
    """用GPT-2生成器和BERT判别器实现的StyleGAN"""
    def __init__(self, author, data_dir="data", output_dir="stylegan_output"):
        """
        初始化StyleGAN
        
        参数:
        - author: 目标作者
        - data_dir: 数据目录
        - output_dir: 输出目录
        """
        self.author = author
        self.data_dir = data_dir
        self.output_dir = os.path.join(output_dir, author)
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 获取所有作者列表
        self.all_authors = [d for d in os.listdir(data_dir) 
                          if os.path.isdir(os.path.join(data_dir, d)) and d != "Unknown"]
        
        logger.info(f"All authors: {', '.join(self.all_authors)}")
        logger.info(f"Target author: {author}")
        
        # 初始化生成器和判别器
        self.generator = GPT2StyleGenerator(
            author=author,
            data_dir=data_dir,
            output_dir=os.path.join(self.output_dir, "generator")
        )
        
        self.discriminator = BERTDiscriminator(
            authors=self.all_authors,
            data_dir=data_dir,
            output_dir=os.path.join(self.output_dir, "discriminator")
        )
    
    def train(self, pretrain_generator=True, train_discriminator=True, 
             gan_epochs=10, num_samples=20):
        """
        训练StyleGAN
        
        参数:
        - pretrain_generator: 是否预训练生成器
        - train_discriminator: 是否训练判别器
        - gan_epochs: GAN训练轮数
        - num_samples: 每轮生成的样本数
        """
        # 步骤1: 预训练生成器
        if pretrain_generator:
            logger.info("Step 1: Pretraining generator...")
            self.generator.train(epochs=3)
            self.generator.generate_samples(num_samples=5, file_prefix="pretrained")
        
        # 步骤2: 训练判别器
        if train_discriminator:
            logger.info("Step 2: Training discriminator...")
            self.discriminator.train(epochs=3)
        
        # 步骤3: GAN训练
        logger.info("Step 3: GAN training...")
        
        # 加载预训练的模型
        self.generator.load_model("best_model")
        self.discriminator.load_model()
        
        # 跟踪最佳样本和最佳模型
        best_samples = []
        best_rewards = []
        best_gan_reward = 0  # 跟踪最佳GAN奖励
        
        for epoch in range(gan_epochs):
            logger.info(f"GAN Epoch {epoch+1}/{gan_epochs}")
            
            # 步骤1: 使用生成器生成样本
            samples = self.generator.generate_samples(
                num_samples=num_samples, 
                file_prefix=f"gan_epoch_{epoch+1}",
                temperature=1.0 - (epoch * 0.05),  # 随着训练进行降低温度
                top_k=50 - (epoch * 2)  # 随着训练进行降低top_k
            )
            
            # 步骤2: 使用判别器评估样本
            sample_rewards = []
            author_probs_list = []
            
            for sample in samples:
                author_probs = self.discriminator.evaluate_text(sample)
                target_prob = author_probs.get(self.author, 0)
                sample_rewards.append(target_prob)
                author_probs_list.append(author_probs)
            
            # 计算当前轮次的平均奖励
            avg_reward = sum(sample_rewards) / len(sample_rewards)
            logger.info(f"Epoch {epoch+1} average reward: {avg_reward:.4f}")
            
            # 如果当前轮次的平均奖励更好，保存模型
            if avg_reward > best_gan_reward:
                best_gan_reward = avg_reward
                self.generator.save_model("gan_best_model")
                logger.info(f"Saved GAN best model with average reward: {avg_reward:.4f}")
            
            # 步骤3: 选择最佳样本
            for i, (sample, reward) in enumerate(zip(samples, sample_rewards)):
                if len(best_samples) < 10 or reward > min(best_rewards):
                    # 评估样本长度和连贯性
                    sample_length = len(sample.split())
                    if sample_length >= 50:  # 确保样本足够长
                        # 添加新样本
                        if len(best_samples) >= 10:
                            # 移除最差样本
                            min_idx = best_rewards.index(min(best_rewards))
                            best_samples.pop(min_idx)
                            best_rewards.pop(min_idx)
                        
                        best_samples.append(sample)
                        best_rewards.append(reward)
                        
                        logger.info(f"New best sample (reward={reward:.4f}):\n{sample[:100]}...")
            
            # 保存当前轮次最佳样本
            with open(os.path.join(self.output_dir, f"best_samples_epoch_{epoch+1}.txt"), "w", encoding="utf-8") as f:
                for i, (sample, reward) in enumerate(zip(best_samples, best_rewards)):
                    f.write(f"Sample {i+1}, Reward: {reward:.4f}\n")
                    f.write(sample)
                    f.write("\n\n" + "-"*50 + "\n\n")
            
            logger.info(f"Epoch {epoch+1} completed. Current best reward: {max(best_rewards):.4f}")
        
        # 保存最终最佳样本
        with open(os.path.join(self.output_dir, "final_best_samples.txt"), "w", encoding="utf-8") as f:
            # 按奖励排序
            sorted_samples = [x for _, x in sorted(zip(best_rewards, best_samples), reverse=True)]
            sorted_rewards = sorted(best_rewards, reverse=True)
            
            for i, (sample, reward) in enumerate(zip(sorted_samples, sorted_rewards)):
                f.write(f"Sample {i+1}, Reward: {reward:.4f}\n")
                f.write(sample)
                f.write("\n\n" + "-"*50 + "\n\n")
        
        logger.info("GAN training completed!")
        logger.info(f"Final best reward: {max(best_rewards):.4f}")
        
        return best_samples, best_rewards

def coherence_evaluation(text, threshold=0.7):
    """
    评估文本连贯性 (简单版本)
    返回0-1之间的分数
    """
    # 1. 检查文本长度
    words = text.split()
    if len(words) < 20:
        return 0.5
    
    # 2. 句子完整性检查
    sentences = sent_tokenize(text)
    complete_sentences = sum(1 for s in sentences if len(s) > 10 and s[-1] in ['.', '!', '?'])
    sentence_score = complete_sentences / max(1, len(sentences))
    
    # 3. 重复性检查
    uniq_words = len(set(words))
    diversity_score = min(1.0, uniq_words / max(1, len(words) * 0.5))
    
    # 4. 标点符号检查
    punct_pattern = re.compile(r'[,.!?;:]')
    punct_count = len(punct_pattern.findall(text))
    expected_punct = len(words) * 0.1  # 大约每10个词一个标点
    punct_score = min(1.0, punct_count / max(1, expected_punct))
    
    # 综合分数
    coherence_score = (sentence_score * 0.5 + diversity_score * 0.3 + punct_score * 0.2)
    return coherence_score

def main():
    # 获取所有作者目录
    data_dir = "data"
    author_dirs = [d for d in os.listdir(data_dir) 
                  if os.path.isdir(os.path.join(data_dir, d)) and d != "Unknown"]
    
    logger.info(f"Available authors: {', '.join(author_dirs)}")
    
    # 为每个作者训练StyleGAN
    for author in author_dirs:
        logger.info(f"Processing author: {author}")
        
        # 创建StyleGAN
        stylegan = StyleGAN(author=author, data_dir=data_dir)
        
        # 训练StyleGAN
        stylegan.train(
            pretrain_generator=True,
            train_discriminator=True,
            gan_epochs=5,
            num_samples=20
        )
    
    logger.info("All authors processed!")

if __name__ == "__main__":
    main()