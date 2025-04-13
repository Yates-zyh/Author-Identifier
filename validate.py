import os
import json
import time
import torch
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
from identify import AuthorIdentifier
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import BertTokenizer
import random
from tqdm import tqdm

# 从train.py中重用函数用于创建滑动窗口样本
def create_sliding_window_samples(text, tokenizer, max_length=512, overlap_tokens=128, max_samples=None):
    """
    从长文本中使用滑动窗口方法创建样本
    
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

def get_samples_from_author_file(file_path, tokenizer, samples_per_file=10, max_length=512, overlap_tokens=128):
    """
    从作者的文件中获取文本样本
    
    参数:
    - file_path: 文件路径
    - tokenizer: 分词器
    - samples_per_file: 每个文件需要的样本数量
    - max_length: 最大token长度
    - overlap_tokens: 滑动窗口重叠token数量
    
    返回:
    - samples: 样本列表
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        # 使用滑动窗口创建样本
        all_samples = create_sliding_window_samples(
            text, 
            tokenizer, 
            max_length=max_length,
            overlap_tokens=overlap_tokens
        )
        
        # 如果样本数量足够，随机选择指定数量的样本
        if len(all_samples) > samples_per_file:
            return random.sample(all_samples, samples_per_file)
        else:
            return all_samples
    except Exception as e:
        print(f"读取文件 {file_path} 时出错: {str(e)}")
        return []

def get_authors_and_samples(data_dir, tokenizer, samples_per_author=30, samples_per_file=10):
    """
    获取数据目录中所有作者的样本
    
    参数:
    - data_dir: 数据目录路径
    - tokenizer: 分词器
    - samples_per_author: On每个作者的总样本数量
    - samples_per_file: 每个文件获取的样本数量
    
    返回:
    - author_samples: 包含作者及其样本的字典
    """
    author_samples = {}
    
    # 获取作者目录列表
    author_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    print(f"在 {data_dir} 中检测到 {len(author_dirs)} 个作者: {', '.join(author_dirs)}")
    
    # 读取每位作者的文件
    for author_name in author_dirs:
        author_path = os.path.join(data_dir, author_name)
        author_all_samples = []
        
        # 获取该作者的所有txt文件
        txt_files = [os.path.join(author_path, f) for f in os.listdir(author_path) if f.endswith('.txt')]
        
        if txt_files:
            print(f"从作者 {author_name} 的 {len(txt_files)} 个文件中收集样本...")
            
            # 从每个文件中获取样本
            for file_path in txt_files:
                file_samples = get_samples_from_author_file(
                    file_path, 
                    tokenizer, 
                    samples_per_file=samples_per_file
                )
                author_all_samples.extend(file_samples)
            
            # 如果样本数量足够，随机选择指定数量的样本
            if len(author_all_samples) > samples_per_author:
                author_samples[author_name] = random.sample(author_all_samples, samples_per_author)
            else:
                author_samples[author_name] = author_all_samples
                
            print(f"作者 {author_name} 共收集了 {len(author_samples[author_name])} 个样本")
    
    return author_samples

def validate_model(model_path, data_dir, samples_per_author=30, samples_per_file=10, save_results=True, plot_confusion_matrix=True):
    """
    使用验证集验证模型性能
    
    参数:
    - model_path: 模型路径
    - data_dir: 验证数据目录
    - samples_per_author: 每个作者的总样本数量
    - samples_per_file: 每个文件获取的样本数量
    - save_results: 是否保存结果
    - plot_confusion_matrix: 是否绘制混淆矩阵
    
    返回:
    - results: 包含验证结果的字典
    """
    print(f"开始验证模型: {model_path}")
    print(f"验证数据目录: {data_dir}")
    
    # 初始化结果字典
    results = {
        "model_path": model_path,
        "data_dir": data_dir,
        "validation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "per_sample_results": [],
        "per_author_results": {},
        "overall_metrics": {}
    }
    
    # 初始化作者识别器
    identifier = AuthorIdentifier(model_path=model_path)
    model_info = identifier.get_model_info()
    results["model_info"] = model_info
    
    # 获取标签名称(作者列表)
    label_names = model_info.get("labels", [])
    print(f"模型可识别的作者: {', '.join(label_names)}")
    
    # 初始化分词器
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    # 获取作者和样本
    author_samples = get_authors_and_samples(
        data_dir, 
        tokenizer, 
        samples_per_author=samples_per_author,
        samples_per_file=samples_per_file
    )
    
    all_true_labels = []
    all_predicted_labels = []
    all_confidence_scores = []
    
    # 对每个作者的每个样本进行验证
    for author_name, samples in author_samples.items():
        author_true_labels = []
        author_predicted_labels = []
        author_confidence_scores = []
        
        # 跳过未知的作者(如果需要)
        if author_name not in label_names and "Unknown Author" not in label_names:
            print(f"警告: 模型未训练识别作者 '{author_name}'，跳过验证...")
            continue
        
        print(f"验证作者 '{author_name}' 的 {len(samples)} 个样本...")
        
        # 使用tqdm显示进度条
        for i, sample_text in enumerate(tqdm(samples, desc=f"验证作者 {author_name} 的样本")):
            sample_id = f"{author_name}_sample_{i+1}"
            
            # 分析样本
            try:
                result = identifier.analyze_text(sample_text)
                
                # 记录结果
                predicted_author = result.get("predicted_author", "Error")
                confidence = result.get("confidence", 0.0)
                
                sample_result = {
                    "sample_id": sample_id,
                    "true_author": author_name,
                    "predicted_author": predicted_author,
                    "confidence": confidence,
                    "is_correct": predicted_author == author_name,
                    "sample_length": len(sample_text.split())  # 记录样本词数
                }
                
                results["per_sample_results"].append(sample_result)
                
                # 为整体指标收集数据
                author_true_labels.append(author_name)
                author_predicted_labels.append(predicted_author)
                author_confidence_scores.append(confidence)
                
            except Exception as e:
                print(f"  分析样本 {sample_id} 时出错: {str(e)}")
                results["per_sample_results"].append({
                    "sample_id": sample_id,
                    "true_author": author_name,
                    "error": str(e)
                })
        
        # 计算每个作者的指标
        if author_true_labels:
            correct_predictions = sum(1 for true, pred in zip(author_true_labels, author_predicted_labels) if true == pred)
            author_accuracy = correct_predictions / len(author_true_labels)
            
            results["per_author_results"][author_name] = {
                "num_samples": len(author_true_labels),
                "num_correct": correct_predictions,
                "accuracy": author_accuracy,
                "avg_confidence": np.mean(author_confidence_scores)
            }
            
            print(f"  作者 '{author_name}' 的准确率: {author_accuracy:.4f} ({correct_predictions}/{len(author_true_labels)})")
            
            # 添加到整体结果
            all_true_labels.extend(author_true_labels)
            all_predicted_labels.extend(author_predicted_labels)
            all_confidence_scores.extend(author_confidence_scores)
    
    # 计算整体指标
    if all_true_labels:
        # 计算整体准确率
        overall_accuracy = accuracy_score(all_true_labels, all_predicted_labels)
        results["overall_metrics"]["accuracy"] = overall_accuracy
        
        # 计算加权F1分数
        unique_labels = sorted(set(all_true_labels) | set(all_predicted_labels))
        f1 = f1_score(all_true_labels, all_predicted_labels, labels=unique_labels, average='weighted')
        results["overall_metrics"]["f1_score"] = f1
        
        # 生成分类报告
        class_report = classification_report(all_true_labels, all_predicted_labels, labels=unique_labels, output_dict=True)
        results["overall_metrics"]["classification_report"] = class_report
        
        # 计算混淆矩阵
        conf_matrix = confusion_matrix(all_true_labels, all_predicted_labels, labels=unique_labels)
        
        # 将混淆矩阵转换为列表(便于JSON序列化)
        results["overall_metrics"]["confusion_matrix"] = conf_matrix.tolist()
        results["overall_metrics"]["confusion_matrix_labels"] = unique_labels
        
        print(f"\n整体准确率: {overall_accuracy:.4f}")
        print(f"加权F1分数: {f1:.4f}")
        
        # 打印分类报告
        print("\n分类报告:")
        print(classification_report(all_true_labels, all_predicted_labels, labels=unique_labels))
        
        # 绘制混淆矩阵
        if plot_confusion_matrix and len(unique_labels) > 1:
            plt.figure(figsize=(10, 8))
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
                       xticklabels=unique_labels, yticklabels=unique_labels)
            plt.xlabel('Prediction')
            plt.ylabel('Ground Truth')
            plt.title('Confusion Matrix')
            
            # 保存图像
            if save_results:
                os.makedirs('visualizations', exist_ok=True)
                plt.savefig('visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
                print("混淆矩阵已保存至 'visualizations/confusion_matrix.png'")
            
            plt.show()
    
    # 保存结果
    if save_results:
        os.makedirs('validation_results', exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        result_path = f'validation_results/validation_{timestamp}.json'
        
        with open(result_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=4)
        
        print(f"验证结果已保存至 '{result_path}'")
    
    return results

def main():
    """
    主函数
    """
    import argparse
    
    # 设置随机种子，确保结果可复现
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)
    
    parser = argparse.ArgumentParser(description='验证作者风格识别模型')
    parser.add_argument('--model_path', type=str, default='../author_style_model', help='模型路径')
    parser.add_argument('--data_dir', type=str, default='data_val', help='验证数据目录')
    parser.add_argument('--samples_per_author', type=int, default=70, help='每个作者的样本数量')
    parser.add_argument('--samples_per_file', type=int, default=70, help='每个文件的样本数量')
    parser.add_argument('--no_save', action='store_false', dest='save_results', help='不保存结果')
    parser.add_argument('--no_plot', action='store_false', dest='plot_confusion_matrix', help='不绘制混淆矩阵')
    
    args = parser.parse_args()
    
    validate_model(
        model_path=args.model_path,
        data_dir=args.data_dir,
        samples_per_author=args.samples_per_author,
        samples_per_file=args.samples_per_file,
        save_results=args.save_results,
        plot_confusion_matrix=args.plot_confusion_matrix
    )

if __name__ == "__main__":
    main()