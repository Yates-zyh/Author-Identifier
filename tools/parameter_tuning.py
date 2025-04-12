import sys
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import json
import time
from tqdm import tqdm
from torch.utils.data import TensorDataset, RandomSampler, SequentialSampler, DataLoader

# 为了导入train.py中的函数，添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train import (
    collect_nltk_samples, 
    prepare_data_from_directory, 
    BertTokenizer, 
    BertForSequenceClassification, 
    AdamW, 
    get_linear_schedule_with_warmup,
    train_model,
    create_sliding_window_samples
)

# 设置设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"使用设备: {device}")

# 定义模型名称
model_name = "bert-base-uncased"
tokenizer = BertTokenizer.from_pretrained(model_name)

def modified_prepare_data(data_dir, nltk_samples, unknown_ratio=1.0, is_training=True, train_sample_ratio=0.7, 
                          pre_extracted_samples=None):
    """
    修改版的数据准备函数，可以控制未知作者样本的比例
    
    参数:
    - data_dir: 包含作家文件夹的目录路径
    - nltk_samples: 预先收集的NLTK样本列表
    - unknown_ratio: 未知作者样本数量的比例 (相对于默认数量的比例)
    - is_training: 是否为训练集
    - train_sample_ratio: 用于训练集的NLTK样本比例
    - pre_extracted_samples: 预先提取的样本字典，包含每个作者的样本和未知样本
    
    返回:
    - dataloader: 数据加载器
    - label_names: 标签名称列表
    - sample_counts: 每个类别的样本数量
    """
    import random
    
    if pre_extracted_samples is None:
        # 如果没有提供预先提取的样本，使用原始方法提取
        # 从train模块获取create_sliding_window_samples函数
        from train import create_sliding_window_samples
        
        all_texts = []
        all_labels = []
        author_samples_dict = {}
        
        # 获取作家目录列表
        author_dirs = [d for d in os.listdir(data_dir) 
                      if os.path.isdir(os.path.join(data_dir, d))]
        print(f"{len(author_dirs)} authors detected: {', '.join(author_dirs)}")
        
        label_names = author_dirs + ["Unknown"]
        
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
                        max_length=512, 
                        overlap_tokens=int(512 * 0.3)
                    )
                    author_texts.extend(file_samples)
                except Exception as e:
                    print(f"  - Unable to read {txt_file}: {str(e)}")
            
            # 存储该作家的所有样本
            author_samples_dict[author_name] = author_texts
            print(f"Author {author_name} has: {len(txt_files)} files / {len(author_texts)} samples")
            
        # 准备未知作者样本
        start_idx = 0 if is_training else int(len(nltk_samples) * train_sample_ratio)
        end_idx = int(len(nltk_samples) * train_sample_ratio) if is_training else len(nltk_samples)
        available_samples = nltk_samples[start_idx:end_idx]
        print(f"Using {len(available_samples)} pre-collected NLTK samples ({start_idx}:{end_idx})")
        
        return_samples = {
            'author_samples': author_samples_dict,
            'unknown_samples': available_samples,
            'label_names': label_names
        }
        
        # 返回提取的样本，同时构建dataloader
        extracted_samples = return_samples
    else:
        # 使用预先提取的样本
        extracted_samples = pre_extracted_samples
        author_samples_dict = extracted_samples['author_samples']
        available_samples = extracted_samples['unknown_samples']
        label_names = extracted_samples['label_names']
    
    # 创建用于当前实验的样本
    all_texts = []
    all_labels = []
    
    # 平衡各作家样本数量
    min_author_samples = min(len(samples) for samples in author_samples_dict.values())
    print(f"Balancing number of samples: Every author will have {min_author_samples} samples")
    
    # 对样本数量进行限制
    balanced_author_dict = {}
    for author_name in author_samples_dict:
        if len(author_samples_dict[author_name]) > min_author_samples:
            balanced_author_dict[author_name] = random.sample(author_samples_dict[author_name], min_author_samples)
        else:
            balanced_author_dict[author_name] = author_samples_dict[author_name]
    
    # 将各作家样本添加到数据集
    for idx, (author_name, samples) in enumerate(balanced_author_dict.items()):
        all_texts.extend(samples)
        all_labels.extend([idx] * len(samples))
    
    # 计算标准未知作者样本数量
    standard_unknown_samples = len(author_samples_dict) * min_author_samples
    
    # 应用比例调整未知作者样本数量
    target_unknown_samples = int(standard_unknown_samples * unknown_ratio)
    print(f"Target number of samples of unknown authors: {target_unknown_samples} "
          f"({unknown_ratio:.2f} of original)")
    
    # 处理未知作者样本
    unknown_samples = []
    if len(available_samples) < target_unknown_samples:
        print(f"Warning: only {len(available_samples)} samples available, using all of them.")
        unknown_samples = available_samples
    else:
        unknown_samples = random.sample(available_samples, target_unknown_samples)
    
    print(f"{len(unknown_samples)} samples selected for unknown authors.")
    
    # 将"未知作家"样本添加到数据集最后一个标签
    unknown_label = len(label_names) - 1  # 最后一个标签是Unknown
    all_texts.extend(unknown_samples)
    all_labels.extend([unknown_label] * len(unknown_samples))
    
    # 打印数据集统计信息
    label_counts = [all_labels.count(i) for i in range(len(label_names))]
    print(f"Number of samples from each author: {label_counts}")
    
    # 保存样本计数
    sample_counts = label_counts
    
    # 分词和编码
    encoded_data = tokenizer(
        all_texts,
        padding='max_length',
        truncation=True,
        max_length=512,
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
    
    if pre_extracted_samples is None:
        return dataloader, label_names, sample_counts, return_samples
    else:
        return dataloader, label_names, sample_counts

def train_and_evaluate(train_dataloader, val_dataloader, label_names, epochs=10, experiment_name="default"):
    """
    训练模型并记录每个epoch的性能指标
    
    参数:
    - train_dataloader: 训练数据加载器
    - val_dataloader: 验证数据加载器
    - label_names: 标签名称列表
    - epochs: 训练轮次
    - experiment_name: 实验名称，用于保存结果
    
    返回:
    - model: 训练好的模型
    - metrics: 包含每轮训练和验证指标的字典
    """
    num_labels = len(label_names)
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_attentions=False,
        output_hidden_states=False,
    )
    model.to(device)
    
    # 创建一个自定义的metric收集函数
    metrics = {
        'train_loss': [],
        'val_loss': [],
        'val_accuracy': [],
        'val_f1': []
    }
    
    # 定义一个回调函数来记录每个epoch的指标
    def epoch_callback(epoch, train_loss, val_accuracy, val_loss, all_preds, all_labels):
        metrics['train_loss'].append(train_loss)
        metrics['val_loss'].append(val_loss)
        metrics['val_accuracy'].append(val_accuracy)
        
        # 计算F1分数
        f1 = f1_score(all_labels, all_preds, average='weighted')
        metrics['val_f1'].append(f1)
        
        print(f"Validation F1 Score: {f1:.4f}")
        
        # 每三轮输出一次详细分类报告或在最后一轮
        if (epoch + 1) % 3 == 0 or epoch == epochs - 1:
            report = classification_report(all_labels, all_preds, target_names=label_names, digits=4)
            print("Classification Report:\n", report)
        
        return True  # 继续训练
    
    # 使用train.py中的train_model函数，但添加我们的回调函数
    model = train_model(model, train_dataloader, val_dataloader, label_names, epochs, epoch_callback=epoch_callback)
    
    # 保存最终模型
    model_save_path = f"../models/author_style_model_{experiment_name}"
    os.makedirs(model_save_path, exist_ok=True)
    model.save_pretrained(model_save_path)
    tokenizer.save_pretrained(model_save_path)
    
    # 保存标签名称
    with open(f"{model_save_path}/label_names.json", 'w') as f:
        json.dump(label_names, f)
    
    # 保存指标
    with open(f"{model_save_path}/metrics.json", 'w') as f:
        json.dump(metrics, f)
    
    # 绘制训练曲线
    plot_training_curves(metrics, experiment_name)
    
    print(f"训练完成！模型和指标已保存到 {model_save_path}")
    return model, metrics

def plot_training_curves(metrics, experiment_name):
    """绘制训练和验证曲线"""
    plt.figure(figsize=(12, 8))
    
    # 绘制损失曲线
    plt.subplot(2, 1, 1)
    plt.plot(metrics['train_loss'], label='Training Loss')
    plt.plot(metrics['val_loss'], label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    # 绘制准确率和F1曲线
    plt.subplot(2, 1, 2)
    plt.plot(metrics['val_accuracy'], label='Validation Accuracy')
    plt.plot(metrics['val_f1'], label='Validation F1')
    plt.title('Validation Metrics')
    plt.xlabel('Epochs')
    plt.ylabel('Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(f"../models/author_style_model_{experiment_name}/training_curves.png")
    plt.close()

def run_experiments():
    """运行一系列实验，测试不同的未知作者样本比例"""
    # 首先收集NLTK样本
    print("收集NLTK样本（只需执行一次）...")
    all_nltk_samples = collect_nltk_samples(tokenizer)
    train_ratio = 0.7  # 70%用于训练，30%用于验证
    
    # 首先提取所有样本
    print("提取所有样本...")
    # 提取训练集样本
    _, _, _, train_extracted_samples = modified_prepare_data(
        data_dir="data_train", 
        nltk_samples=all_nltk_samples,
        unknown_ratio=1.0,  # 提取全部样本
        is_training=True,
        train_sample_ratio=train_ratio
    )
    
    # 提取验证集样本
    _, _, _, val_extracted_samples = modified_prepare_data(
        data_dir="data_val", 
        nltk_samples=all_nltk_samples,
        unknown_ratio=1.0,  # 提取全部样本
        is_training=False,
        train_sample_ratio=train_ratio
    )
    
    # 设置不同的未知作者样本比例
    unknown_ratios = [1.0, 0.7, 0.5, 0.3, 0.1]
    results = {}
    
    for ratio in unknown_ratios:
        print(f"\n\n===== 实验：未知作者样本比例 = {ratio} =====")
        experiment_name = f"unknown_ratio_{ratio:.1f}"
        
        # 准备数据 - 使用正确的数据路径和预提取的样本
        train_dataloader, label_names_train, train_counts = modified_prepare_data(
            data_dir="data_train", 
            nltk_samples=all_nltk_samples,
            unknown_ratio=ratio,
            is_training=True,
            train_sample_ratio=train_ratio,
            pre_extracted_samples=train_extracted_samples
        )
        
        val_dataloader, label_names_val, val_counts = modified_prepare_data(
            data_dir="data_val", 
            nltk_samples=all_nltk_samples,
            unknown_ratio=ratio,
            is_training=False,
            train_sample_ratio=train_ratio,
            pre_extracted_samples=val_extracted_samples
        )
        
        # 打印样本统计信息
        print(f"训练集样本分布: {train_counts}")
        print(f"验证集样本分布: {val_counts}")
        
        # 训练和评估模型
        _, metrics = train_and_evaluate(
            train_dataloader, 
            val_dataloader, 
            label_names_train, 
            epochs=10,
            experiment_name=experiment_name
        )
        
        # 保存结果
        results[ratio] = {
            'final_accuracy': metrics['val_accuracy'][-1],
            'final_f1': metrics['val_f1'][-1],
            'best_accuracy': max(metrics['val_accuracy']),
            'best_f1': max(metrics['val_f1']),
            'best_epoch_accuracy': metrics['val_accuracy'].index(max(metrics['val_accuracy'])) + 1,
            'best_epoch_f1': metrics['val_f1'].index(max(metrics['val_f1'])) + 1
        }
    
    # 比较不同比例的结果
    compare_results(results)

def compare_results(results):
    """比较不同实验的结果并生成报告"""
    # 创建结果DataFrame
    df = pd.DataFrame(results).T
    df.index.name = 'unknown_ratio'
    df.reset_index(inplace=True)
    
    # 保存结果到CSV
    results_path = "../models/unknown_ratio_experiment_results.csv"
    df.to_csv(results_path, index=False)
    
    # 绘制比较图
    plt.figure(figsize=(12, 6))
    
    # 准确率比较
    plt.subplot(1, 2, 1)
    plt.plot(df['unknown_ratio'], df['best_accuracy'], 'o-', label='Best Accuracy')
    plt.plot(df['unknown_ratio'], df['final_accuracy'], 's--', label='Final Accuracy')
    for i, row in df.iterrows():
        plt.annotate(f"{row['best_epoch_accuracy']}", 
                    (row['unknown_ratio'], row['best_accuracy']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    plt.title('Accuracy vs Unknown Ratio')
    plt.xlabel('Unknown Sample Ratio')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)
    
    # F1分数比较
    plt.subplot(1, 2, 2)
    plt.plot(df['unknown_ratio'], df['best_f1'], 'o-', label='Best F1')
    plt.plot(df['unknown_ratio'], df['final_f1'], 's--', label='Final F1')
    for i, row in df.iterrows():
        plt.annotate(f"{row['best_epoch_f1']}", 
                    (row['unknown_ratio'], row['best_f1']),
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    plt.title('F1 Score vs Unknown Ratio')
    plt.xlabel('Unknown Sample Ratio')
    plt.ylabel('F1 Score')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig("../models/unknown_ratio_comparison.png")
    
    print(f"结果已保存到 {results_path}")
    print("最优未知作者样本比例:")
    best_acc_row = df.loc[df['best_accuracy'].idxmax()]
    best_f1_row = df.loc[df['best_f1'].idxmax()]
    print(f"- 最高准确率: {best_acc_row['best_accuracy']:.4f} (比例={best_acc_row['unknown_ratio']}, 轮次={int(best_acc_row['best_epoch_accuracy'])})")
    print(f"- 最高F1分数: {best_f1_row['best_f1']:.4f} (比例={best_f1_row['unknown_ratio']}, 轮次={int(best_f1_row['best_epoch_f1'])})")

if __name__ == "__main__":
    # 创建保存模型的目录
    os.makedirs("../models", exist_ok=True)
    
    # 运行所有实验
    run_experiments()