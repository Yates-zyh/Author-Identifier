import os
import argparse
from pdfminer.high_level import extract_text

def pdf_to_txt(pdf_path, output_path=None):
    """
    将PDF文件转换为TXT文件
    
    参数:
        pdf_path (str): PDF文件的路径
        output_path (str, optional): 输出TXT文件的路径。如果未提供，将使用与PDF相同的名称但扩展名为.txt
    
    返回:
        str: 生成的TXT文件路径
    """
    # 如果未提供输出路径，使用与输入相同的文件名但扩展名为.txt
    if output_path is None:
        output_path = os.path.splitext(pdf_path)[0] + '.txt'
    
    try:
        # 从PDF中提取文本
        text = extract_text(pdf_path)
        
        # 将文本写入TXT文件
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        
        print(f"转换成功: {pdf_path} → {output_path}")
        return output_path
    
    except Exception as e:
        print(f"转换过程中出错: {e}")
        return None

def main():
    import sys
    # 检查是否有命令行参数
    if len(sys.argv) > 1:
        # 创建命令行参数解析器
        parser = argparse.ArgumentParser(description='将PDF文件转换为TXT格式')
        parser.add_argument('pdf_path', help='PDF文件路径')
        parser.add_argument('-o', '--output', help='输出TXT文件路径 (可选)')
        
        # 解析命令行参数
        args = parser.parse_args()
        
        # 执行转换
        pdf_to_txt(args.pdf_path, args.output)
    else:
        # 直接转换指定的PDF文件
        pdf_path = r"E:\NUSCourses\EBA5004 Practival Language Processing\PM\project\data\Márquez\In_Evil_Hour.pdf"
        pdf_to_txt(pdf_path)

if __name__ == "__main__":
    main()
