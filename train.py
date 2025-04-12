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
import matplotlib.pyplot as plt  # 添加matplotlib导入
import os  # 添加os模块导入

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

# 添加一个新函数，专门用于收集NLTK语料库的样本
def collect_nltk_samples(tokenizer, max_length=512, overlap_percent=0.3):
    """
    从NLTK语料库中收集样本，仅需调用一次
    
    参数:
    - tokenizer: 分词器
    - max_length: 文本最大token长度
    - overlap_percent: 滑动窗口重叠比例
    
    返回:
    - nltk_samples: 从NLTK语料库中收集的样本列表
    """
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
    
    print(f"{len(all_corpus_samples)} total samples collected from NLTK corpus libraries.")
    return all_corpus_samples

# 修改数据准备函数，支持传入预先收集的NLTK样本
def prepare_data_from_directory(data_dir, max_length=512, overlap_percent=0.2, 
                                balance_samples=True, is_training=True, nltk_samples=None,
                                train_sample_ratio=0.7):
    """
    从目录结构中准备多个作家的文本数据，并生成用于训练或验证的数据集
    
    参数:
    - data_dir: 包含作家文件夹的目录路径
    - max_length: 文本最大token长度
    - overlap_percent: 滑动窗口重叠比例
    - balance_samples: 是否平衡各类别的样本数量
    - is_training: 是否为训练集（影响数据加载器的创建方式）
    - nltk_samples: 预先收集的NLTK样本列表（如果提供，则不会重新收集）
    - train_sample_ratio: 用于训练集的NLTK样本比例 (0-1之间)
    """
    import os
    
    all_texts = []
    all_labels = []
    author_samples_dict = {}  # 用于存储每位作家的样本
    
    # 获取作家目录列表
    author_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"{len(author_dirs)} authors detected in {data_dir}: {', '.join(author_dirs)}")
    
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
    
    # 计算需要的"未知作家"样本数量
    target_unknown_samples = int((len(author_samples_dict) / 2) * min_author_samples)
    print(f"Target number of samples of unknown authors: {target_unknown_samples} (occupy half of total samples)")
    
    # 使用NLTK样本
    unknown_samples = []
    start_idx = 0 if is_training else int(len(nltk_samples) * train_sample_ratio)
    end_idx = int(len(nltk_samples) * train_sample_ratio) if is_training else len(nltk_samples)
    available_samples = nltk_samples[start_idx:end_idx]
        
    print(f"Using {len(available_samples)} pre-collected NLTK samples ({start_idx}:{end_idx})")
        
    if len(available_samples) < target_unknown_samples:
        print(f"Warning: only {len(available_samples)} samples available, using all of them.")
        unknown_samples = available_samples
    else:
        unknown_samples = random.sample(available_samples, int(target_unknown_samples))
    
    print(f"{len(unknown_samples)} samples selected for unknown authors.")
    
    # 将"未知作家"样本添加到数据集最后一个标签
    unknown_label = len(author_dirs) 
    all_texts.extend(unknown_samples)
    all_labels.extend([unknown_label] * len(unknown_samples))
    
    # 打印数据集统计信息
    label_counts = [all_labels.count(i) for i in range(len(label_names))]
    print(f"Number of samples from each author: {label_counts}")
    print(f"Total number of samples: {len(all_texts)}")
    
    # 分词和编码
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
    
    # 创建DataLoader
    batch_size = 8
    
    dataset = TensorDataset(input_ids, attention_masks, labels)
    
    if is_training:
        sampler = RandomSampler(dataset)
    else:
        sampler = SequentialSampler(dataset)
        
    dataloader = DataLoader(dataset, sampler=sampler, batch_size=batch_size)
    
    return dataloader, label_names

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

# 添加新函数用于可视化训练历史
def visualize_training_history(history, save_path=None):
    """
    可视化训练历史
    
    参数:
    - history: 包含训练历史数据的字典
    - save_path: 可选的保存路径
    """
    epochs = history.get('epochs', [])
    if not epochs:
        print("No training history to visualize")
        return
        
    # 创建一个新的图形
    plt.figure(figsize=(15, 10))
    
    # 绘制损失曲线
    plt.subplot(2, 2, 1)
    plt.plot(epochs, history['train_loss'], 'b-', label='Training Loss')
    plt.plot(epochs, history['val_loss'], 'r-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率曲线
    plt.subplot(2, 2, 2)
    plt.plot(epochs, history['val_accuracy'], 'g-', label='Validation Accuracy')
    plt.title('Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # 绘制F1分数曲线（如果有的话）
    if 'f1_score' in history:
        plt.subplot(2, 2, 3)
        plt.plot(epochs, history['f1_score'], 'm-', label='F1 Score')
        plt.title('F1 Score')
        plt.xlabel('Epochs')
        plt.ylabel('F1 Score')
        plt.legend()
        plt.grid(True)
    
    # 判断是否过拟合
    if len(epochs) > 1:
        is_overfitting = False
        overfitting_epoch = None
        
        for i in range(1, len(epochs)):
            if (history['val_loss'][i] > history['val_loss'][i-1] and 
                history['train_loss'][i] < history['train_loss'][i-1]):
                is_overfitting = True
                overfitting_epoch = epochs[i]
                break
        
        plt.subplot(2, 2, 4)
        plt.text(0.5, 0.5, 
                f"过拟合检测:\n{'可能在第 ' + str(overfitting_epoch) + ' 轮开始过拟合' if is_overfitting else '未检测到明显过拟合'}\n\n"
                f"过拟合解决方案:\n"
                f"1. 增加训练数据量\n"
                f"2. 添加正则化（如dropout、权重衰减）\n"
                f"3. 减小模型复杂度\n"
                f"4. 早停（Early Stopping）\n"
                f"5. 数据增强",
                ha='center', va='center', fontsize=12)
        plt.axis('off')
    
    plt.tight_layout()
    
    # 如果提供了保存路径，则保存图像
    if save_path:
        plt.savefig(save_path)
        print(f"Training history visualization saved to {save_path}")
    
    plt.show()

def train_model(model, train_dataloader, val_dataloader, label_names, epochs, epoch_callback=None):
    # 创建一个字典来存储训练历史
    history = {
        'epochs': [],
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'f1_score': []
    }
    
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
        from sklearn.metrics import classification_report, confusion_matrix, f1_score
        report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
        conf_matrix = confusion_matrix(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        print(f"Accuracy: {avg_val_accuracy:.4f}")
        print(f"Loss: {avg_val_loss:.4f}")
        print(f"F1 Score: {f1:.4f}")
        print("Classification Report:\n", report)
        print("Confusion Matrix:\n", conf_matrix)
        
        # 添加本轮结果到历史记录
        history['epochs'].append(epoch + 1)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_accuracy'].append(avg_val_accuracy)
        history['f1_score'].append(f1)
        
        # 调用回调函数（如果提供）
        if epoch_callback:
            continue_training = epoch_callback(
                epoch=epoch,
                train_loss=avg_train_loss,
                val_accuracy=avg_val_accuracy,
                val_loss=avg_val_loss,
                all_preds=all_preds,
                all_labels=all_labels
            )
            if not continue_training:
                print("Early stopping triggered by callback.")
                break
    
    print("Training done!")
    
    # 可视化训练历史
    if not os.path.exists('visualizations'):
        os.makedirs('visualizations')
    visualize_training_history(history, save_path='visualizations/training_history.png')
    
    return model, history

def main():
    print("Collecting NLTK samples (this will be done only once)...")
    all_nltk_samples = collect_nltk_samples(tokenizer)
    train_ratio = 0.7  # 70%用于训练，30%用于验证

    # 数据准备（使用预先收集的NLTK样本）
    train_dataloader, label_names_train = prepare_data_from_directory(
        data_dir="data_train", 
        balance_samples=True, 
        is_training=True,
        nltk_samples=all_nltk_samples,
        train_sample_ratio=train_ratio
    )

    val_dataloader, label_names_val = prepare_data_from_directory(
        data_dir="data_val", 
        balance_samples=True, 
        is_training=False,
        nltk_samples=all_nltk_samples,
        train_sample_ratio=train_ratio
    )

    # 使用训练集的标签
    label_names = label_names_train
    num_labels = len(label_names)
    print(f"Number of labels: {num_labels}")

    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    fine_tuned_model, training_history = train_model(model, train_dataloader, val_dataloader, label_names, epochs=20)

    # 保存模型和标签名称
    model_save_path = "../author_style_model"
    fine_tuned_model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)

    # 保存标签名称
    import json
    with open(f"{model_save_path}/label_names.json", 'w') as f:
        json.dump(label_names, f)
        
    print(f"Authors' styles saved to {model_save_path}")

if __name__ == "__main__":
    main()