import os
import sys
import re

def remove_non_content_header(lines):
    """识别并移除文本开头的非正文内容"""
    # Gutenberg标记，通常表示正文开始
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** BEGIN OF THIS PROJECT GUTENBERG EBOOK",
        "***BEGIN THE PROJECT GUTENBERG EBOOK"
    ]
    
    # 章节开始标记
    chapter_patterns = [
        r"^CHAPTER [IVXLCivxlc]+",  # CHAPTER I, CHAPTER II 等
        r"^Chapter \d+",            # Chapter 1, Chapter 2 等
        r"^\d+\.\s",                # 1. 2. 等
        r"^BOOK [IVXLCivxlc]+",     # BOOK I, BOOK II 等
    ]
    
    # 查找正文开始的位置
    content_start_idx = None
    
    # 1. 首先尝试使用Gutenberg的标准标记
    for i, line in enumerate(lines):
        for marker in start_markers:
            if marker in line:
                # 找到了开始标记，正文通常在标记后几行开始
                content_start_idx = i + 3  # 跳过标记行及空行
                break
        if content_start_idx:
            break
    
    # 2. 如果没找到标准标记，尝试寻找第一个章节开始
    if not content_start_idx:
        for i, line in enumerate(lines):
            if i > 15:  # 只检查前几十行，避免把正文中的章节标记误认为是开始
                for pattern in chapter_patterns:
                    if re.match(pattern, line.strip()):
                        content_start_idx = i
                        break
            if content_start_idx:
                break
    
    # 3. 如果仍未找到，尝试查找第一个较长的段落（很可能是正文）
    if not content_start_idx:
        for i, line in enumerate(lines):
            if i > 20 and len(line.strip()) > 100:  # 长段落通常是正文
                content_start_idx = i
                break
    
    # 4. 如果以上方法都失败，默认保留前30行后的内容
    if not content_start_idx and len(lines) > 30:
        content_start_idx = 30
    
    # 如果找到了内容起始点，则从那里开始返回
    if content_start_idx and content_start_idx < len(lines):
        return lines[content_start_idx:]
    
    # 否则返回原始内容
    return lines

def remove_non_content_footer(lines):
    """识别并移除文本结尾的非正文内容"""
    # Gutenberg结束标记
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG",
        "***THE END***",
        "*** END ***"
    ]
    
    # 其他通用的结束标记
    generic_end_markers = [
        "THE END",
        "The End",
        "the end",
        "END.",
        "End.",
        "FIN",
        "- THE END -"
    ]
    
    content_end_idx = None
    
    # 1. 首先寻找Gutenberg的标准结束标记
    for i, line in enumerate(lines):
        for marker in end_markers:
            if marker in line:
                content_end_idx = i
                break
        if content_end_idx is not None:
            break
    
    # 2. 如果没找到标准标记，寻找通用结束标记
    if content_end_idx is None:
        # 从文件的后四分之一开始搜索
        start_search_idx = max(0, int(len(lines) * 0.75))
        for i in range(start_search_idx, len(lines)):
            line = lines[i].strip()
            if line in generic_end_markers:
                # 确认这是真正的结束标记（通常是单独一行且在文件末尾附近）
                content_end_idx = i
                break
    
    # 如果找到了结束点，则只返回到那个点之前的内容
    if content_end_idx is not None:
        return lines[:content_end_idx]
    
    # 否则返回原始内容
    return lines

def remove_empty_lines(file_path, clean_header=True, clean_footer=True):
    """读取文件，删除空行和非正文部分，然后保存回原文件"""
    try:
        # 读取文件内容
        with open(file_path, 'r', encoding='utf-8', errors='replace') as file:
            lines = file.readlines()
        
        # 首先移除非正文头部（如果启用）
        if clean_header:
            lines = remove_non_content_header(lines)
        
        # 移除非正文尾部（如果启用）
        if clean_footer:
            lines = remove_non_content_footer(lines)
        
        # 过滤掉空行（包括只有空格的行）
        non_empty_lines = [line for line in lines if line.strip()]
        
        # 写回文件
        with open(file_path, 'w', encoding='utf-8') as file:
            file.writelines(non_empty_lines)
        
        return True
    
    except Exception as e:
        print(f"处理文件 {file_path} 时出错: {e}")
        return False

def process_directory(root_dir, clean_header=True, clean_footer=True):
    """遍历目录及其子目录，处理所有txt文件"""
    processed_count = 0
    error_count = 0
    
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.txt'):
                file_path = os.path.join(root, file)
                print(f"正在处理: {file_path}")
                
                if remove_empty_lines(file_path, clean_header, clean_footer):
                    processed_count += 1
                else:
                    error_count += 1
    
    return processed_count, error_count

if __name__ == "__main__":
    # 设置gutenberg_novels目录路径
    novels_dir = r"e:\NUSCourses\EBA5004 Practival Language Processing\PM\project\gutenberg_novels"
    
    if not os.path.exists(novels_dir):
        print(f"错误: 目录 '{novels_dir}' 不存在!")
        sys.exit(1)
    
    # 默认启用头部和尾部清理
    clean_header = True
    clean_footer = True
    
    # 检查命令行参数
    if len(sys.argv) > 1:
        if "no-header-clean" in sys.argv:
            clean_header = False
            print("头部清理功能已禁用")
        if "no-footer-clean" in sys.argv:
            clean_footer = False
            print("尾部清理功能已禁用")
    
    print(f"开始处理 {novels_dir} 目录中的所有txt文件...")
    processed, errors = process_directory(novels_dir, clean_header, clean_footer)
    
    print("\n处理完成!")
    print(f"成功处理: {processed} 个文件")
    print(f"处理失败: {errors} 个文件")
