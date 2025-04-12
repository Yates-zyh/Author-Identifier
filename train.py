import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
import numpy as np
from tqdm import tqdm
import random
import nltk
from nltk.corpus import gutenberg, brown, reuters, webtext
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

# 从语料库中收集样本的通用函数
def collect_samples_from_corpus(corpus, fileids, max_samples, tokenizer, max_length, overlap_percent):
    """
    Collect samples from the specified corpus
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
    Collect samples from NLTK corpus, only needs to be called once
    
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
    Prepare text data from multiple authors from a directory structure and generate datasets for training or validation
    
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
    Create samples from long text using sliding window method
    
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

def train_model(model, train_dataloader, val_dataloader, label_names, epochs, 
            learning_rate=2e-5, weight_decay=0.01, 
            save_checkpoint=True, checkpoint_dir='checkpoints'):
    """
    Train the model and evaluate performance
    
    参数:
    - model: 待训练的模型
    - train_dataloader: 训练数据加载器
    - val_dataloader: 验证数据加载器
    - label_names: 标签名称列表
    - epochs: 训练轮次
    - learning_rate: 学习率
    - weight_decay: 权重衰减参数
    - save_checkpoint: 是否保存检查点
    - checkpoint_dir: 检查点保存目录
    
    返回:
    - model: 训练后的模型
    """
    # 准备优化器和学习率调度器
    optimizer = AdamW(model.parameters(), lr=learning_rate, eps=1e-8, weight_decay=weight_decay)
    total_steps = len(train_dataloader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=int(total_steps * 0.1),  # 10% 的步骤用于热身
        num_training_steps=total_steps
    )
    
    # 早停设置
    early_stopping_counter = 0
    best_val_loss = float('inf')
    best_model_state = None
    
    # 确保检查点目录存在
    if save_checkpoint and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    for epoch in range(epochs):
        print(f'======== Epoch {epoch + 1} / {epochs} ========')
        
        # 训练阶段
        model.train()
        total_train_loss = 0
        for batch in tqdm(train_dataloader, desc="Training"):
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
        print(f"Average training loss: {avg_train_loss:.4f}")
        
        # 评估阶段
        model.eval()
        total_eval_accuracy = 0
        total_eval_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(val_dataloader, desc="Evaluating"):
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
        
        print(f"Validation Accuracy: {avg_val_accuracy:.4f}")
        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"F1 Score: {f1:.4f}")
        
        # 检查是否需要保存最佳模型
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = model.state_dict().copy()
            early_stopping_counter = 0
            
            # 保存检查点
            if save_checkpoint:
                checkpoint_path = os.path.join(checkpoint_dir, f'best_model_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_loss': avg_val_loss,
                    'val_accuracy': avg_val_accuracy,
                    'f1_score': f1,
                    'label_names': label_names,
                }, checkpoint_path)
    
    # 如果存在最佳模型状态，则加载它
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model

def main(train_dir, val_dir, model_save_dir, 
         model_name, epochs, balance_samples,
         learning_rate, weight_decay):
    """
    Main function for training author identification model
    
    参数:
    - train_dir: 训练数据目录
    - val_dir: 验证数据目录
    - model_save_dir: 模型保存目录
    - model_name: 预训练模型名称
    - epochs: 训练轮次
    - balance_samples: 是否平衡各类别样本数
    - learning_rate: 学习率
    - weight_decay: 权重衰减参数
    """
    # 设置随机种子，确保结果可复现
    seed_val = 42
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    
    print(f"Starting author identification model training process...")
    print(f"Training data directory: {train_dir}")
    print(f"Validation data directory: {val_dir}")
    print(f"Model will be saved to: {model_save_dir}")
    print(f"Pre-trained model: {model_name}")
    print(f"Training epochs: {epochs}")
    print(f"Device: {device}")
    
    # 确保模型保存目录存在
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    # 收集NLTK样本（仅需执行一次）
    all_nltk_samples = collect_nltk_samples(tokenizer)
    train_ratio = 0.7  # 70%用于训练，30%用于验证

    # 数据准备（使用预先收集的NLTK样本）
    train_dataloader, label_names_train = prepare_data_from_directory(
        data_dir=train_dir, 
        balance_samples=balance_samples, 
        is_training=True,
        nltk_samples=all_nltk_samples,
        train_sample_ratio=train_ratio
    )

    val_dataloader, label_names_val = prepare_data_from_directory(
        data_dir=val_dir, 
        balance_samples=balance_samples, 
        is_training=False,
        nltk_samples=all_nltk_samples,
        train_sample_ratio=train_ratio
    )

    # 验证标签一致性
    if set(label_names_train) != set(label_names_val):
        combined_labels = list(set(label_names_train) | set(label_names_val))
        label_names = sorted(combined_labels)
    else:
        label_names = label_names_train
        
    num_labels = len(label_names)

    # 初始化模型
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)

    # 训练模型
    fine_tuned_model, training_history = train_model(
        model, 
        train_dataloader, 
        val_dataloader, 
        label_names, 
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        save_checkpoint=True,
        checkpoint_dir=os.path.join(model_save_dir, 'checkpoints')
    )

    # 保存最终模型和标签名称
    model_timestamp = time.strftime("%Y%m%d_%H%M%S")
    final_model_dir = os.path.join(model_save_dir, f"author_model_{model_timestamp}")
    
    # 保存模型和分词器
    fine_tuned_model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    # 保存标签名称
    import json
    with open(os.path.join(final_model_dir, "label_names.json"), 'w', encoding='utf-8') as f:
        json.dump(label_names, f, ensure_ascii=False, indent=4)
    
    # 保存模型元数据
    metadata = {
        "model_name": model_name,
        "training_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "num_labels": num_labels,
        "training_epochs": epochs,
        "final_accuracy": training_history['val_accuracy'][-1] if training_history['val_accuracy'] else None,
        "final_f1_score": training_history['f1_score'][-1] if training_history['f1_score'] else None,
        "label_names": label_names,
        "train_data_dir": train_dir,
        "val_data_dir": val_dir
    }
    
    with open(os.path.join(final_model_dir, "model_metadata.json"), 'w', encoding='utf-8') as f:
        json.dump(metadata, f, ensure_ascii=False, indent=4)
        
    return fine_tuned_model, label_names, final_model_dir

if __name__ == "__main__":
    # 可以通过命令行参数覆盖默认配置
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Train author style identification model')
    parser.add_argument('--train_dir', type=str, default='data_train', help='Training data directory')
    parser.add_argument('--val_dir', type=str, default='data_val', help='Validation data directory')
    parser.add_argument('--model_dir', type=str, default='../author_style_model', help='Model save directory')
    parser.add_argument('--model_name', type=str, default='bert-base-uncased', help='Pre-trained model name')
    parser.add_argument('--epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--no_balance', action='store_false', dest='balance_samples', help='Do not balance samples across classes')
    parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.01, help='Weight decay parameter')
    
    args = parser.parse_args()
    
    main(
        train_dir=args.train_dir,
        val_dir=args.val_dir,
        model_save_dir=args.model_dir,
        model_name=args.model_name,
        epochs=args.epochs,
        balance_samples=args.balance_samples,
        learning_rate=args.lr,
        weight_decay=args.weight_decay
    )