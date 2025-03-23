import re
import os

def clean_text_file(input_file, output_file=None):
    """
    清理文本文件，删除单独占一行的数字和页标
    
    Args:
        input_file (str): 输入文件路径
        output_file (str, optional): 输出文件路径，如果不提供则覆盖原文件
    """
    if output_file is None:
        output_file = input_file
    
    # 读取文件内容
    with open(input_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 删除单独占一行的数字
    content = re.sub(r'^\s*\d+\s*$', '', content, flags=re.MULTILINE)
    
    # 删除包含页标的行
    page_header = "The Moon and Sixpence"
    content = re.sub(r'^.*' + re.escape(page_header) + r'.*$', '', content, flags=re.MULTILINE)
    
    # 清理多余空行
    content = re.sub(r'\n\s*\n', '\n\n', content)
    
    # 写入处理后的内容
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(content)
    
    print(f"处理完成！")
    print(f"已清理文件: {input_file}")
    if input_file != output_file:
        print(f"结果已保存至: {output_file}")

if __name__ == "__main__":
    input_file_path = "e:\\NUSCourses\\EBA5004 Practival Language Processing\\PM\\author_works2.txt"
    
    # 如果你希望创建一个新文件而不是覆盖原文件，请取消下面一行的注释并修改路径
    # output_file_path = "e:\\NUSCourses\\EBA5004 Practival Language Processing\\PM\\author_works_cleaned.txt"
    # clean_text_file(input_file_path, output_file_path)
    
    # 直接覆盖原文件
    clean_text_file(input_file_path)
