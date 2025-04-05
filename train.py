import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import nltk
from nltk.corpus import gutenberg, brown, reuters, webtext

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 从语料库中收集样本的通用函数
def collect_samples_from_corpus(corpus, fileids, max_samples, tokenizer, max_length, overlap_percent):
    """
    从指定的语料库中收集样本
    """
    samples = []
    for fileid in fileids:
        text = corpus.raw(fileid)
        samples.extend(create_sliding_window_samples(
            text,
            tokenizer,
            max_length=max_length,
            overlap_tokens=int(max_length * overlap_percent),
            max_samples=max_samples
        ))
    return samples

# 修改数据准备函数，支持从目录结构中读取多个作家的作品
def prepare_data_from_directory(data_dir="data", max_length=512, overlap_percent=0.3, balance_samples=True):
    """
    从目录结构中准备多个作家的文本数据，并生成用于训练的数据集
    
    参数:
    - data_dir: 包含作家文件夹的目录路径
    - max_length: 文本最大token长度
    - overlap_percent: 滑动窗口重叠比例
    - balance_samples: 是否平衡各类别的样本数量
    """
    import os
    
    all_texts = []
    all_labels = []
    author_samples_dict = {}  # 用于存储每位作家的样本
    
    # 获取作家目录列表
    author_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"{len(author_dirs)} authors detected: {', '.join(author_dirs)}")
    
    label_names = author_dirs + ["Unknown"]  # 添加"未知作家"标签
    
    # 读取每位作家的文本
    for idx, author_name in enumerate(author_dirs):
        author_path = os.path.join(data_dir, author_name)
        author_texts = []
        
        # 获取该作家的所有txt文件
        txt_files = [f for f in os.listdir(author_path) if f.endswith('.txt')]
        
        # 读取每个文件并添加到作家文本列表
        for txt_file in txt_files:
            file_path = os.path.join(author_path, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    file_text = f.read()
                    
                # 对每个文件内容创建样本
                file_samples = create_sliding_window_samples(
                    file_text, 
                    tokenizer, 
                    max_length=max_length, 
                    overlap_tokens=int(max_length * overlap_percent)
                )
                author_texts.extend(file_samples)
            except Exception as e:
                print(f"  - Unable to read {txt_file}: {str(e)}")
        
        # 存储该作家的所有样本
        author_samples_dict[author_name] = author_texts
        print(f"Author {author_name} has: {len(txt_files)} files / {len(author_texts)} samples")
    
    # 平衡各作家样本数量
    if balance_samples and author_samples_dict:
        min_author_samples = min(len(samples) for samples in author_samples_dict.values())
        print(f"Balancing number of samples: Every author will have {min_author_samples} samples")
        
        # 对样本数量进行限制
        for author_name in author_samples_dict:
            if len(author_samples_dict[author_name]) > min_author_samples:
                author_samples_dict[author_name] = random.sample(author_samples_dict[author_name], min_author_samples)
    
    # 将各作家样本添加到数据集
    for idx, (author_name, samples) in enumerate(author_samples_dict.items()):
        all_texts.extend(samples)
        all_labels.extend([idx] * len(samples))
    
    # 计算需要的"未知作家"样本数量（等于所有已知作家样本总和，占总样本数的一半）
    target_unknown_samples = len(author_samples_dict) * min_author_samples
    print(f"Target number of samples of unknown authors: {target_unknown_samples} (occupy half of total samples)")
    
    # 生成"未知作家"样本 - 使用多个NLTK语料库
    unknown_samples = []
    
    # 确保NLTK语料库已下载
    corpus_list = ['gutenberg', 'brown', 'reuters', 'webtext']
    for corpus_name in corpus_list:
        try:
            nltk.data.find(f'corpora/{corpus_name}')
        except LookupError:
            print(f"Downloading NLTK {corpus_name}...")
            nltk.download(corpus_name)
            print(f"NLTK {corpus_name} downloaded.")

    # 从所有语料库收集样本
    corpus_configs = [
        ("Gutenberg", gutenberg, gutenberg.fileids(), 50),
        ("Brown", brown, brown.fileids(), 25),
        ("Reuters", reuters, reuters.fileids(), 15),
        ("Webtext", webtext, webtext.fileids(), 10)
    ]
    all_corpus_samples = []
    for corpus_name, corpus, fileids, max_samples in corpus_configs:
        print(f"Collecting samples from {corpus_name}...")
        all_corpus_samples.extend(collect_samples_from_corpus(
            corpus, fileids, max_samples, tokenizer, max_length, overlap_percent
        ))
    print(f"{len(all_corpus_samples)} samples collected from NLTK.")
    
    # 随机抽样以获取目标数量的未知作家样本
    if len(all_corpus_samples) < target_unknown_samples:
        print(f"Warning: only {len(all_corpus_samples)} samples available, using all of them.")
        unknown_samples = all_corpus_samples
    else:
        unknown_samples = random.sample(all_corpus_samples, target_unknown_samples)
    print(f"{len(unknown_samples)} samples selected for unknown authors.")
    
    # 将"未知作家"样本添加到数据集
    unknown_label = len(author_dirs)  # 最后一个标签是"未知作家"
    all_texts.extend(unknown_samples)
    all_labels.extend([unknown_label] * len(unknown_samples))
    
    # 打印数据集统计信息
    label_counts = [all_labels.count(i) for i in range(len(label_names))]
    print(f"Number of samples from each author: {label_counts}")
    print(f"Total number of samples: {len(all_texts)}")
    
    # 对文本进行分词和编码
    encoded_data = tokenizer(
        all_texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    # 创建数据集
    input_ids = encoded_data['input_ids']
    attention_masks = encoded_data['attention_mask']
    labels = torch.tensor(all_labels)
    
    # 分割为训练集和验证集
    train_inputs, val_inputs, train_masks, val_masks, train_labels, val_labels = train_test_split(
        input_ids, attention_masks, labels, test_size=0.15, random_state=42, stratify=labels
    )
    
    # 创建DataLoaders
    batch_size = 8
    
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    
    return train_dataloader, val_dataloader, label_names

# 添加一个新函数，用于创建滑动窗口样本
def create_sliding_window_samples(text, tokenizer, max_length, overlap_tokens, max_samples=None):
    """
    使用滑动窗口方法从长文本中创建样本
    
    参数:
    - text: 输入文本
    - tokenizer: 分词器
    - max_length: 窗口的最大token长度
    - overlap_tokens: 相邻窗口之间重叠的token数量
    - max_samples: 最大样本数量限制
    
    返回:
    - samples: 文本样本列表
    """
    # 清理文本，移除多余空白符
    text = ' '.join(text.split())
    
    # 对整个文本进行分词，获取tokens
    tokens = tokenizer.encode(text)
    
    samples = []
    start_idx = 0
    
    # 使用滑动窗口切分文本
    while start_idx < len(tokens):
        # 确保不超出文本长度
        end_idx = min(start_idx + max_length, len(tokens))
        
        window_tokens = tokens[start_idx:end_idx]
        
        if len(window_tokens) >= 100:
            window_text = tokenizer.decode(window_tokens)
            samples.append(window_text)
        
        # 如果已经到达文本末尾，退出循环
        if end_idx == len(tokens):
            break
            
        # 更新下一个窗口的起始位置（考虑重叠）
        start_idx += (max_length - overlap_tokens)
        
        # 如果达到样本数量限制，提前结束
        if max_samples and len(samples) >= max_samples:
            break
    
    return samples

def train_model(train_dataloader, val_dataloader, label_names, epochs):
    optimizer = AdamW(model.parameters(), lr=2e-5, eps=1e-8, weight_decay=0.01)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),  # 10% 的步骤用于热身
        num_training_steps=total_steps
    )
    
    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader):
            model.zero_grad()
            
            input_ids = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)
            
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_train_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
        
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"Average training loss: {avg_train_loss}")
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                input_ids = batch[0].to(device)
                attention_mask = batch[1].to(device)
                labels = batch[2].to(device)
                
                outputs = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                
                loss = outputs.loss
                total_eval_loss += loss.item()
                
                logits = outputs.logits
                predictions = torch.argmax(logits, dim=1)
                
                # 收集预测和标签用于计算F1分数
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                
                accuracy = (predictions == labels).float().mean().item()
                total_eval_accuracy += accuracy
        
        avg_val_accuracy = total_eval_accuracy / len(val_dataloader)
        avg_val_loss = total_eval_loss / len(val_dataloader)
        
        # 计算F1分数和其他指标
        from sklearn.metrics import classification_report, confusion_matrix
        report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        
        print(f"Accuracy: {avg_val_accuracy:.4f}")
        print(f"Loss: {avg_val_loss:.4f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
    
    print("Training done!")
    return model

# 数据准备
train_dataloader, val_dataloader, label_names = prepare_data_from_directory(data_dir="data_train", balance_samples=True)
num_labels = len(label_names)
print(f"Number of labels: {num_labels}")
model = BertForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    output_attentions=False,
    output_hidden_states=False,
)
model.to(device)

fine_tuned_model = train_model(train_dataloader, val_dataloader, label_names, epochs=3)

# 保存模型和标签名称
model_save_path = "../author_style_model"
model.save_pretrained(model_save_path)
tokenizer.save_pretrained(model_save_path)

# 保存标签名称
import json
with open(f"{model_save_path}/label_names.json", 'w') as f:
    json.dump(label_names, f)
    
print(f"Authors' styles saved to {model_save_path}")