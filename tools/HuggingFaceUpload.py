from huggingface_hub import HfApi, create_repo
import os
import dotenv
import shutil

# 加载环境变量
dotenv.load_dotenv()

# 从环境变量获取 Hugging Face token
GENERATION_TOKEN = os.environ.get("GENERATION_TOKEN")  
IDENTIFICATION_TOKEN = os.environ.get("IDENTIFICATION_TOKEN")

# 确保 token 已设置
if not GENERATION_TOKEN:
    raise ValueError("请确保在 .env 文件中设置了 GENERATION_TOKEN 环境变量")
if not IDENTIFICATION_TOKEN:
    raise ValueError("请确保在 .env 文件中设置了 IDENTIFICATION_TOKEN 环境变量")

# 设置环境变量
os.environ["GENERATION_TOKEN"] = GENERATION_TOKEN
os.environ["IDENTIFICATION_TOKEN"] = IDENTIFICATION_TOKEN

# 新仓库信息
new_username = "Yates-zyh"
new_model_name = "author_identifier"
new_repo_name = f"{new_username}/{new_model_name}"

api = HfApi()

# 创建新仓库
create_repo(new_repo_name, exist_ok=True, token=IDENTIFICATION_TOKEN)

# 上传author_style_model文件夹到新仓库
model_path = "author_style_model"
api.upload_folder(
    folder_path=model_path,
    repo_id=new_repo_name,
    repo_type="model",
    path_in_repo="",  # 上传到根目录
    token=IDENTIFICATION_TOKEN
)

# 创建模型说明文档
model_card = """
---
language: en
license: mit
tags:
- text-classification
- author-identification
datasets:
- custom
---

# Author Identifier Model

This repository contains a fine-tuned BERT model trained to identify the writing style of various authors.

## Available Authors

- Agatha Christie
- Alexandre Dumas
- Arthur Conan Doyle
- Charles Dickens
- Charlotte Brontë
- F. Scott Fitzgerald
- García Márquez
- Herman Melville
- Jane Austen
- Mark Twain

## Model Details

- **Model type:** BERT for sequence classification
- **Training data:** Custom dataset of authors' works
- **Authors:** Yates-zyh

## Usage

```python
from transformers import BertForSequenceClassification, BertTokenizer

# 加载模型
model = BertForSequenceClassification.from_pretrained("Yates-zyh/author_identifier")
tokenizer = BertTokenizer.from_pretrained("Yates-zyh/author_identifier")

# 使用模型进行预测
text = "Your text sample here"
inputs = tokenizer(text, return_tensors="pt", padding='max_length', truncation=True, max_length=512)
outputs = model(**inputs)
```
"""

api.upload_file(
    path_or_fileobj=model_card.encode(),
    path_in_repo="README.md",
    repo_id=new_repo_name,
    token=IDENTIFICATION_TOKEN
)

print(f"模型已成功上传到 Hugging Face Hub: {new_repo_name}")