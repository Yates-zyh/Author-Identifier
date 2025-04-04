# 安装crawl4ai库 网页：https://github.com/unclecode/crawl4ai
# pip install -U crawl4ai
# playwright install
import asyncio
import re
import os
import random
from crawl4ai import *

async def get_book_links(crawler, author_url):
    """获取作者页面中的前n本书籍链接，此处n为10"""
    result = await crawler.arun(url=author_url)
    content = result.html
    n = 10
    
    # 提取取前n个书籍链接，古腾堡项目通常将书籍链接格式化为/ebooks/数字
    book_links = re.findall(r'<a[^>]*href="(/ebooks/\d+)"[^>]*>', content)
    unique_links = []
    seen = set()
    for link in book_links:
        if link not in seen:
            seen.add(link)
            unique_links.append(link)
        if len(unique_links) >= n:
            break
    return ["https://www.gutenberg.org" + link for link in unique_links]

async def get_read_online_link(crawler, book_url):
    """跳转网页阅读链接"""
    result = await crawler.arun(url=book_url)
    content = result.html
    
    specific_links = re.findall(r'<a[^>]*href="(/ebooks/\d+\.html\.images)"[^>]*>', content)
    if specific_links:
        link = specific_links[0]
        if not link.startswith('http'):
            link = "https://www.gutenberg.org" + link
        return link
    return None

async def process_book(crawler, book_url):
    """处理单本书籍"""
    read_online_url = await get_read_online_link(crawler, book_url)
    result = await crawler.arun(url=read_online_url)
    content = result.markdown
    
    # 提取标题和作者
    title = re.search(r'\*\*Title\*\*\s*:\s*([^\n]+)', content).group(1).strip()
    author = re.search(r'\*\*Author\*\*\s*:\s*([^\n]+)', content).group(1).strip()
    
    folder_path = os.path.join(os.path.dirname(__file__), author.replace(' ', '_'))
    os.makedirs(folder_path, exist_ok=True)
    
    # 清理内容
    clean_content = re.sub(r'##\s+[IVXLCDM]+\s*\n', '', content)
    clean_content = re.sub(r'##\s+[^\n]+\n', '', clean_content)
    clean_content = re.sub(r'#\s+[^\n]+\n', '', clean_content)
    clean_content = re.sub(r'^by\s+[^\n]+\n', '', clean_content, 
                        flags=re.IGNORECASE | re.MULTILINE)
    
    # 保存txt文件
    safe_filename = re.sub(r'[\\/*?:"<>|]', '', title) + ".txt"
    file_path = os.path.join(folder_path, safe_filename)
    with open(file_path, 'w', encoding='utf-8') as f:
        f.write(clean_content.strip())
    print(f"内容已保存到 {file_path} 文件")
    
    return title, author

async def main():
    """
    37: 狄更斯
    65: 莎士比亚
    68: 简·奥斯汀
    85: 雨果
    125: 约瑟夫·康拉德
    126: 毛姆
    251: 巴尔扎克
    408: 夏洛特·勃朗特
    600: 惠特曼
    634: 欧亨利"""
    author_url = "https://www.gutenberg.org/ebooks/author/600"
    
    async with AsyncWebCrawler() as crawler:
        # 获取作者的前十本书链接
        book_links = await get_book_links(crawler, author_url)
        print(f"找到 {len(book_links)} 本书籍")
        
        # 处理每本书
        successful_count = 0
        for i, book_url in enumerate(book_links):
            print(f"正在处理第 {i+1} 本书: {book_url}")
            # 添加短暂延迟避免请求过于频繁
            await asyncio.sleep(random.uniform(1, 3))
            try:
                result = await process_book(crawler, book_url)
                if result:
                    title, author = result
                    print(f"成功处理: {title} by {author}")
                    successful_count += 1
            except Exception as e:
                print(f"处理书籍 {book_url} 时发生错误: {e}")
        
        print(f"总共成功处理 {successful_count} 本书")

if __name__ == "__main__":
    asyncio.run(main())