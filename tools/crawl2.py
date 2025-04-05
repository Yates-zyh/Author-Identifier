import os
import requests
from bs4 import BeautifulSoup
import time
import random
import re

# 自定义Crawler类代替crawl4ai
class Crawler:
    def __init__(self, user_agent=None, respect_robots_txt=True, delay=1.0):
        self.user_agent = user_agent
        self.respect_robots_txt = respect_robots_txt
        self.delay = delay
        self.last_request_time = 0
        self.session = requests.Session()
        if user_agent:
            self.session.headers.update({'User-Agent': user_agent})
    
    def get(self, url):
        # 实现请求间隔
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        if time_since_last < self.delay:
            time.sleep(self.delay - time_since_last)
        
        response = self.session.get(url)
        self.last_request_time = time.time()
        return response

# 定义20位流行小说家
popular_authors = [
    "Jane Austen", "Charles Dickens", "Mark Twain", "Leo Tolstoy", 
    "Fyodor Dostoevsky", "Oscar Wilde", "Edgar Allan Poe", "Virginia Woolf",
    "George Orwell", "Ernest Hemingway", "F. Scott Fitzgerald", "Herman Melville",
    "Jules Verne", "H.G. Wells", "Arthur Conan Doyle", "Agatha Christie",
    "Victor Hugo", "Alexandre Dumas", "Emily Bronte", "Charlotte Bronte"
]

# 创建基础保存目录
base_dir = "e:/NUSCourses/EBA5004 Practival Language Processing/PM/project/gutenberg_novels"
os.makedirs(base_dir, exist_ok=True)

# 初始化爬虫
crawler = Crawler(
    user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    respect_robots_txt=True,
    delay=2.0  # 请求间隔时间(秒)
)

# Gutenberg网站URL
base_url = "https://www.gutenberg.org"

# 为每位作家爬取小说
for author in popular_authors:
    # 创建作家目录
    author_dir = os.path.join(base_dir, author.replace(" ", "_"))
    os.makedirs(author_dir, exist_ok=True)
    
    print(f"正在爬取{author}的小说...")
    
    # 构建搜索URL - 搜索作者名并筛选英语作品
    search_query = author.replace(" ", "+")
    search_url = f"{base_url}/ebooks/search/?query={search_query}&submit_search=Go%21&language=en"
    
    try:
        # 获取搜索结果页面
        search_page = crawler.get(search_url)
        soup = BeautifulSoup(search_page.content, "html.parser")
        
        # 找到所有书籍链接
        books = []
        for book_element in soup.select(".booklink"):
            title_element = book_element.select_one(".title")
            if title_element:
                title = title_element.text.strip()
                link = book_element.select_one("a")
                if link and link.has_attr("href"):
                    book_url = f"{base_url}{link['href']}"
                    books.append((title, book_url))
        
        # 只保留前10本，如果不足则全部保留
        books = books[:10]
        
        # 处理每本书
        for idx, (title, book_url) in enumerate(books, 1):
            try:
                print(f"  [{idx}/{len(books)}] 正在处理《{title}》...")
                
                # 访问书籍详情页
                book_page = crawler.get(book_url)
                book_soup = BeautifulSoup(book_page.content, "html.parser")
                
                # 寻找纯文本下载链接
                text_link = None
                for file_format in book_soup.select(".files a"):
                    if "Plain Text" in file_format.text:
                        text_link = file_format["href"]
                        break
                
                if not text_link:
                    print(f"    未找到《{title}》的纯文本链接")
                    continue
                
                # 获取完整的下载链接
                if not text_link.startswith("http"):
                    text_link = f"{base_url}{text_link}"
                
                # 下载文本内容
                text_response = crawler.get(text_link)
                
                # 处理文件名，移除非法字符
                safe_title = re.sub(r'[\\/*?:"<>|]', "_", title)
                file_path = os.path.join(author_dir, f"{safe_title}.txt")
                
                # 保存文件
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(text_response.text)
                
                print(f"    已保存至 {file_path}")
                
            except Exception as e:
                print(f"    处理《{title}》时出错: {str(e)}")
                continue
        
        print(f"已完成{author}的爬取工作。共处理了{len(books)}本小说。")
        
    except Exception as e:
        print(f"搜索{author}时出错: {str(e)}")
    
    # 作家之间添加额外延迟，避免请求过于频繁
    time.sleep(random.uniform(3, 5))

print("爬取工作全部完成!")
