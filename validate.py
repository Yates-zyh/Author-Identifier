import os
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
from identify import analyze_text_style

def load_validation_data(val_data_dir, samples_per_author, max_text_length=512):
    """
    从验证数据目录加载文本段落用于测试
    
    参数:
    - val_data_dir: 验证数据目录
    - samples_per_author: 每位作者取出的样本数
    - max_text_length: 每个样本的最大文本长度
    
    返回:
    - samples: 文本样本列表
    - true_authors: 真实作者列表
    - file_sources: 文件来源
    """
    samples = []
    true_authors = []
    file_sources = []
    
    # 读取模型支持的作者列表
    model_path = "../author_style_model"
    with open(f"{model_path}/label_names.json", 'r') as f:
        label_names = json.load(f)
    
    # 过滤掉"Unknown"标签
    known_authors = [author for author in label_names if author != "Unknown"]
    
    # 获取作家目录列表
    author_dirs = [d for d in os.listdir(val_data_dir) 
                if os.path.isdir(os.path.join(val_data_dir, d)) and d in known_authors]
    
    print(f"在验证数据中找到 {len(author_dirs)} 位作者: {', '.join(author_dirs)}")
    
    for author_name in author_dirs:
        author_path = os.path.join(val_data_dir, author_name)
        
        # 获取该作家的所有txt文件
        txt_files = [f for f in os.listdir(author_path) if f.endswith('.txt')]
        
        author_samples = []
        author_files = []
        
        # 从每个文件中读取段落
        for txt_file in txt_files:
            file_path = os.path.join(author_path, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 简单分段，每段最多max_text_length个字符
                paragraphs = []
                words = text.split()
                current_paragraph = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > max_text_length:
                        if current_paragraph:  # 确保段落不为空
                            paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = [word]
                        current_length = len(word)
                    else:
                        current_paragraph.append(word)
                        current_length += len(word) + 1  # +1 for space
                
                if current_paragraph:  # 添加最后一个段落
                    paragraphs.append(' '.join(current_paragraph))
                
                # 过滤掉太短的段落
                paragraphs = [p for p in paragraphs if len(p.split()) >= 50]
                
                # 如果段落太多，随机选择几个
                if len(paragraphs) > 80:
                    selected_paragraphs = random.sample(paragraphs, 80)
                else:
                    selected_paragraphs = paragraphs
                
                author_samples.extend(selected_paragraphs)
                author_files.extend([txt_file] * len(selected_paragraphs))
                
            except Exception as e:
                print(f"  - 无法读取 {txt_file}: {str(e)}")
        
        # 如果样本数量超过指定值，随机选择
        if len(author_samples) > samples_per_author:
            indices = random.sample(range(len(author_samples)), samples_per_author)
            selected_samples = [author_samples[i] for i in indices]
            selected_files = [author_files[i] for i in indices]
        else:
            selected_samples = author_samples
            selected_files = author_files
        
        # 添加到总列表
        samples.extend(selected_samples)
        true_authors.extend([author_name] * len(selected_samples))
        file_sources.extend(selected_files)
        
        print(f"作者 {author_name}: 添加了 {len(selected_samples)} 个样本段落")
    
    return samples, true_authors, file_sources

def main():
    # 数据路径
    val_data_dir = "data_val"
    
    # 加载验证数据
    print("加载验证数据...")
    samples, true_authors, file_sources = load_validation_data(
        val_data_dir=val_data_dir,
        samples_per_author=80  # 每位作者取80个样本
    )
    
    print(f"总共加载了 {len(samples)} 个样本用于验证")
    
    # 预测结果
    print("进行预测分析...")
    results = []
    predicted_authors = []
    confidence_scores = []
    
    for sample in tqdm(samples):
        result = analyze_text_style(sample)
        results.append(result)
        predicted_authors.append(result["predicted_author"])
        confidence_scores.append(result["confidence"])
    
    # 计算准确率
    correct = sum(p == t for p, t in zip(predicted_authors, true_authors))
    accuracy = correct / len(true_authors) if true_authors else 0
    
    print(f"\n总体准确率: {accuracy:.4f} ({correct}/{len(true_authors)})")
    
    # 按作者评估准确率
    print("\n按作者的准确率:")
    unique_authors = set(true_authors)
    
    for author in unique_authors:
        author_indices = [i for i, a in enumerate(true_authors) if a == author]
        author_correct = sum(predicted_authors[i] == true_authors[i] for i in author_indices)
        author_accuracy = author_correct / len(author_indices)
        
        print(f"  {author}: {author_accuracy:.4f} ({author_correct}/{len(author_indices)})")
    
    # 混淆矩阵和分类报告
    unique_labels = sorted(list(set(true_authors + predicted_authors)))
    
    print("\n分类报告:")
    print(classification_report(true_authors, predicted_authors, labels=unique_labels, digits=4))
    
    # print("\n混淆矩阵:")
    # cm = confusion_matrix(true_authors, predicted_authors, labels=unique_labels)
    
    # # 打印带标签的混淆矩阵
    # print("    " + " ".join(f"{label[:7]:>7}" for label in unique_labels))
    # for i, row in enumerate(cm):
    #     print(f"{unique_labels[i][:7]:>7} " + " ".join(f"{cell:>7}" for cell in row))
    
if __name__ == "__main__":
    main()

import os
import random
from tqdm import tqdm
from sklearn.metrics import classification_report, confusion_matrix
import json
import numpy as np
from identify import analyze_text_style

def load_validation_data(val_data_dir, samples_per_author, max_text_length=512):
    """
    从验证数据目录加载文本段落用于测试
    
    参数:
    - val_data_dir: 验证数据目录
    - samples_per_author: 每位作者取出的样本数
    - max_text_length: 每个样本的最大文本长度
    
    返回:
    - samples: 文本样本列表
    - true_authors: 真实作者列表
    - file_sources: 文件来源
    """
    samples = []
    true_authors = []
    file_sources = []
    
    # 读取模型支持的作者列表
    model_path = "../author_style_model"
    with open(f"{model_path}/label_names.json", 'r') as f:
        label_names = json.load(f)
    
    # 过滤掉"Unknown"标签
    known_authors = [author for author in label_names if author != "Unknown"]
    
    # 获取作家目录列表
    author_dirs = [d for d in os.listdir(val_data_dir) 
                if os.path.isdir(os.path.join(val_data_dir, d)) and d in known_authors]
    
    print(f"在验证数据中找到 {len(author_dirs)} 位作者: {', '.join(author_dirs)}")
    
    for author_name in author_dirs:
        author_path = os.path.join(val_data_dir, author_name)
        
        # 获取该作家的所有txt文件
        txt_files = [f for f in os.listdir(author_path) if f.endswith('.txt')]
        
        author_samples = []
        author_files = []
        
        # 从每个文件中读取段落
        for txt_file in txt_files:
            file_path = os.path.join(author_path, txt_file)
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    text = f.read()
                
                # 简单分段，每段最多max_text_length个字符
                paragraphs = []
                words = text.split()
                current_paragraph = []
                current_length = 0
                
                for word in words:
                    if current_length + len(word) + 1 > max_text_length:
                        if current_paragraph:  # 确保段落不为空
                            paragraphs.append(' '.join(current_paragraph))
                        current_paragraph = [word]
                        current_length = len(word)
                    else:
                        current_paragraph.append(word)
                        current_length += len(word) + 1  # +1 for space
                
                if current_paragraph:  # 添加最后一个段落
                    paragraphs.append(' '.join(current_paragraph))
                
                # 过滤掉太短的段落
                paragraphs = [p for p in paragraphs if len(p.split()) >= 50]
                
                # 如果段落太多，随机选择几个
                if len(paragraphs) > 20:
                    selected_paragraphs = random.sample(paragraphs, 20)
                else:
                    selected_paragraphs = paragraphs
                
                author_samples.extend(selected_paragraphs)
                author_files.extend([txt_file] * len(selected_paragraphs))
                
            except Exception as e:
                print(f"  - 无法读取 {txt_file}: {str(e)}")
        
        # 如果样本数量超过指定值，随机选择
        if len(author_samples) > samples_per_author:
            indices = random.sample(range(len(author_samples)), samples_per_author)
            selected_samples = [author_samples[i] for i in indices]
            selected_files = [author_files[i] for i in indices]
        else:
            selected_samples = author_samples
            selected_files = author_files
        
        # 添加到总列表
        samples.extend(selected_samples)
        true_authors.extend([author_name] * len(selected_samples))
        file_sources.extend(selected_files)
        
        print(f"作者 {author_name}: 添加了 {len(selected_samples)} 个样本段落")
    
    return samples, true_authors, file_sources

def main():
    # 数据路径
    val_data_dir = "data_val"
    
    # 加载验证数据
    print("加载验证数据...")
    samples, true_authors, file_sources = load_validation_data(
        val_data_dir=val_data_dir,
        samples_per_author=20  # 每位作者取20个样本
    )
    
    print(f"总共加载了 {len(samples)} 个样本用于验证")
    
    # 预测结果
    print("进行预测分析...")
    results = []
    predicted_authors = []
    confidence_scores = []
    
    for sample in tqdm(samples):
        result = analyze_text_style(sample)
        results.append(result)
        predicted_authors.append(result["predicted_author"])
        confidence_scores.append(result["confidence"])
    
    # 计算准确率
    correct = sum(p == t for p, t in zip(predicted_authors, true_authors))
    accuracy = correct / len(true_authors) if true_authors else 0
    
    print(f"\n总体准确率: {accuracy:.4f} ({correct}/{len(true_authors)})")
    
    # 按作者评估准确率
    print("\n按作者的准确率:")
    unique_authors = set(true_authors)
    
    for author in unique_authors:
        author_indices = [i for i, a in enumerate(true_authors) if a == author]
        author_correct = sum(predicted_authors[i] == true_authors[i] for i in author_indices)
        author_accuracy = author_correct / len(author_indices)
        
        print(f"  {author}: {author_accuracy:.4f} ({author_correct}/{len(author_indices)})")
    
    # 混淆矩阵和分类报告
    unique_labels = sorted(list(set(true_authors + predicted_authors)))
    
    print("\n分类报告:")
    print(classification_report(true_authors, predicted_authors, labels=unique_labels, digits=4))
    
    # print("\n混淆矩阵:")
    # cm = confusion_matrix(true_authors, predicted_authors, labels=unique_labels)
    
    # # 打印带标签的混淆矩阵
    # print("    " + " ".join(f"{label[:7]:>7}" for label in unique_labels))
    # for i, row in enumerate(cm):
    #     print(f"{unique_labels[i][:7]:>7} " + " ".join(f"{cell:>7}" for cell in row))
    
if __name__ == "__main__":
    main()
