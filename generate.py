import argparse
import json
import torch
import os
import numpy as np
from transformers import BertTokenizer, BertForSequenceClassification, GPT2LMHeadModel, GPT2Tokenizer
from tqdm import tqdm

def load_author_model(model_path):
    """
    加载作家风格识别模型和标签
    """
    # 加载标签名称
    with open(os.path.join(model_path, "label_names.json"), 'r') as f:
        label_names = json.load(f)
    
    # 加载模型和tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    
    return model, tokenizer, label_names

def generate_text_with_style(author, word_count, model_path="../author_style_model", device=None):
    """
    根据指定作家的风格生成文本
    
    参数:
    - author: 作家名称
    - word_count: 要生成的大致字数
    - model_path: 模型路径
    - device: 设备(cpu/cuda)
    
    返回:
    - 生成的文本
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print(f"使用设备: {device}")
    
    # 加载作家风格模型
    author_model, author_tokenizer, label_names = load_author_model(model_path)
    author_model.to(device)
    author_model.eval()
    
    # 检查作家名称是否在标签中
    if author not in label_names:
        valid_authors = [name for name in label_names if name != "Unknown"]
        print(f"作家'{author}'不在已知作家列表中。")
        print(f"可用作家: {', '.join(valid_authors)}")
        return None
    
    author_id = label_names.index(author)
    print(f"已选择作家: {author} (ID: {author_id})")
    
    # 加载GPT-2模型用于文本生成
    gpt2_model = GPT2LMHeadModel.from_pretrained("gpt2")
    gpt2_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    gpt2_tokenizer.pad_token = gpt2_tokenizer.eos_token
    gpt2_model.to(device)
    gpt2_model.eval()
    
    # 设置生成参数
    max_length = 250  # 段落最大长度
    temperature = 0.7  # 降低温度以提高连贯性
    top_k = 50
    top_p = 0.92
    repetition_penalty = 1.2
    
    # 计算需要生成的段落数量
    approx_words_per_paragraph = 100
    num_paragraphs = max(1, word_count // approx_words_per_paragraph)
    
    generated_paragraphs = []
    retry_limit = 10  # 增加每个段落的最大尝试次数(原来是5)
    
    # 设置作家风格的提示
    current_prompt = f"In the style of {author}:"
    
    print(f"开始生成大约 {word_count} 个字的文本，风格类似于 {author}...")
    
    # 最终生成的全文
    full_text = ""
    
    # 生成每个段落
    for i in tqdm(range(num_paragraphs), desc="生成段落"):
        best_paragraph = None
        best_score = -float('inf')
        scores = []
        
        # 尝试生成符合作家风格的段落
        for attempt in range(retry_limit):
            # 生成文本，添加作家风格提示
            input_ids = gpt2_tokenizer.encode(
                current_prompt, 
                return_tensors="pt"
            ).to(device)
            
            # 创建正确的attention mask
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                output = gpt2_model.generate(
                    input_ids,
                    attention_mask=attention_mask,  # 添加attention mask
                    max_length=max_length,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    do_sample=True,
                    num_return_sequences=1,
                    pad_token_id=gpt2_tokenizer.eos_token_id
                )
            
            paragraph = gpt2_tokenizer.decode(output[0], skip_special_tokens=True)
            
            # 移除可能重复的提示部分
            if paragraph.startswith(current_prompt):
                paragraph = paragraph[len(current_prompt):].strip()
            
            # 使用作家风格模型评估段落
            encoded_para = author_tokenizer(
                paragraph,
                padding='max_length',
                truncation=True,
                max_length=512,
                return_tensors='pt'
            ).to(device)
            
            with torch.no_grad():
                outputs = author_model(
                    input_ids=encoded_para['input_ids'],
                    attention_mask=encoded_para['attention_mask']
                )
                
                logits = outputs.logits
                probs = torch.nn.functional.softmax(logits, dim=1)
                author_score = probs[0][author_id].item()
            
            scores.append(author_score)
            
            # 如果这个段落的得分更高，则保留
            if author_score > best_score:
                best_score = author_score
                best_paragraph = paragraph
                
                # 动态阈值策略: 随着尝试次数增加，降低接受阈值
                accept_threshold = 0.8 - (attempt * 0.05)  # 每次尝试降低阈值0.05
                accept_threshold = max(0.5, accept_threshold)  # 但不低于0.5
                
                if author_score > accept_threshold:
                    break
        
        # 如果所有尝试都未能产生高分段落，选择得分最高的
        if best_paragraph is None and scores:
            best_score_index = np.argmax(scores)
            best_paragraph = paragraph  # 使用最后生成的段落，此处可能需要修正
        
        # 添加生成的段落到全文
        if best_paragraph:
            if full_text:  # 如果不是第一段，添加段落分隔符
                full_text += "\n\n"
            full_text += best_paragraph
            
            # 将当前段落的后半部分设为下一段的开头提示，增加连贯性
            words = best_paragraph.split()
            if len(words) > 15:  # 如果段落足够长
                # 取最后10-15个词作为下一段的提示
                continuation_words = words[-min(15, len(words)//3):]
                current_prompt = " ".join(continuation_words)
            else:
                # 如果段落太短，使用整个段落
                current_prompt = best_paragraph
            
            # 每隔一段打印状态
            if (i+1) % 5 == 0:
                print(f"已生成 {i+1}/{num_paragraphs} 段落")
                print(f"最近段落得分: {best_score:.4f}")
    
    # 计算实际生成的字数
    actual_word_count = len(full_text.split())
    print(f"已生成 {actual_word_count} 个单词的文本")
    
    return full_text

def main():
    parser = argparse.ArgumentParser(description="根据特定作家风格生成文本")
    parser.add_argument("--author", type=str, required=True, help="作家名称")
    parser.add_argument("--words", type=int, default=500, help="要生成的大致字数")
    parser.add_argument("--model_path", type=str, default="../author_style_model", help="模型路径")
    parser.add_argument("--output", type=str, help="输出文件路径（可选）")
    
    args = parser.parse_args()
    
    generated_text = generate_text_with_style(
        author=args.author,
        word_count=args.words,
        model_path=args.model_path
    )
    
    if generated_text:
        print("\n生成的文本:")
        print("="*80)
        print(generated_text)
        print("="*80)

if __name__ == "__main__":
    main()
