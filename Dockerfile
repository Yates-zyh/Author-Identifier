FROM python:3.9-slim

WORKDIR /app

# 复制应用代码
COPY . /app/

# 安装依赖
RUN pip install --no-cache-dir streamlit torch transformers numpy pandas matplotlib

# 创建模型目录
RUN mkdir -p /app/author_style_model

# 设置环境变量
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=true
ENV STREAMLIT_SERVER_ADDRESS="0.0.0.0"
ENV STREAMLIT_SERVER_BASE_URL_PATH="author-identifier"

# 暴露端口
EXPOSE 8501

# 启动命令
CMD ["streamlit", "run", "GUI_streamlit.py"]
