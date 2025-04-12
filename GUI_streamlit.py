"""
使用以下命令启动本地 Web 应用程序：
streamlit run GUI_streamlit.py
启动 Streamlit 后，请在浏览器中访问：http://127.0.0.1:8501
"""
import streamlit as st
from identify import AuthorIdentifier
import os
import torch
import time
from datetime import datetime

# 导入chatbot_with_generator中的类和函数
from chatbot_with_generator import AuthorStyleAPI, chat_with_deepseek

# 为Streamlit环境扩展AuthorStyleAPI功能
class StreamlitAuthorStyleAPI(AuthorStyleAPI):
    """扩展AuthorStyleAPI以适应Streamlit界面"""
    
    def _wait_for_rate_limit(self):
        """确保请求间隔符合速率限制"""
        if self.last_request_time is not None:
            elapsed = (datetime.now() - self.last_request_time).total_seconds()
            if elapsed < self.min_request_interval:
                time.sleep(self.min_request_interval - elapsed)
        self.last_request_time = datetime.now()

    def _load_model(self, author, max_retries=3):
        """加载指定作者的模型，带重试机制"""
        if author in self.loaded_models:
            return self.loaded_models[author]

        for attempt in range(max_retries):
            try:
                self._wait_for_rate_limit()
                with st.spinner(f"Loading {author}'s model... (Attempt {attempt + 1}/{max_retries})"):
                    # 调用父类方法加载模型
                    return super()._load_model(author)
            except Exception as e:
                if "429" in str(e):  # 速率限制错误
                    wait_time = (attempt + 1) * 10  # 等待时间指数增长
                    st.warning(f"API rate limit reached. Waiting {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    st.error(f"Error loading {author}'s model: {str(e)}")
                    if attempt == max_retries - 1:
                        raise

    def generate_best_sample(self, author, num_samples=3, max_length=200):
        """生成多个样本并返回评分最高的一个"""
        with st.spinner(f"Generating {num_samples} samples to find the best text..."):
            samples = self.generate_text(author, num_samples, max_length)
            best_sample = None
            best_score = -1

            progress_bar = st.progress(0)
            for i, sample in enumerate(samples):
                try:
                    progress_bar.progress((i+1)/len(samples))
                    score = self.evaluate_text(sample, author)
                    if score > best_score:
                        best_score = score
                        best_sample = sample
                except Exception as e:
                    st.error(f"Error evaluating sample: {str(e)}")
                    continue

            return best_sample, best_score

# Streamlit版本的DeepSeek聊天函数
def streamlit_chat_with_deepseek(author):
    with st.spinner(f"Generating text in {author}'s style using DeepSeek..."):
        return chat_with_deepseek(author)

def main():
    # 设置页面标题和配置
    st.set_page_config(page_title="Author Style Tool", layout="wide")
    
    # 创建包含分析工具、生成工具和关于信息的标签页
    tab1, tab2, tab3 = st.tabs(["Style Analysis", "Style Generation", "About"])
    
    with tab1:
        # 页面标题
        st.title("Author Style Identifier")
        st.markdown("This tool can analyze text and identify potential author styles.")
        
        # 模型信息显示（仅初始化一次）
        if 'model_info' not in st.session_state:
            with st.spinner("Loading model information..."):
                try:
                    identifier = AuthorIdentifier()
                    st.session_state.model_info = identifier.get_model_info()
                    st.session_state.identifier = identifier
                except Exception as e:
                    st.error(f"Failed to load model: {str(e)}")
                    return
        
        # 显示模型信息
        with st.expander("Model Information", expanded=False):
            model_info = st.session_state.model_info
            st.markdown(f"**Model Path**: {model_info.get('model_path', 'Unknown')}")
            st.markdown(f"**Device**: {model_info.get('device', 'Unknown')}")
            st.markdown(f"**Supported Authors**: {', '.join(model_info.get('labels', ['Unknown']))}")
            st.markdown(f"**Training Date**: {model_info.get('training_date', 'Unknown')}")
        
        # 文本输入区域
        col1, col2 = st.columns([3, 1])
        with col1:
            text_input = st.text_area(
                "Enter text to analyze:", 
                height=200, 
                help="Paste the text you want to analyze here."
            )
        
        with col2:
            # 高级设置
            st.subheader("Analysis Settings")
            confidence_threshold = st.slider(
                "Confidence Threshold", 
                min_value=0.0, 
                max_value=1.0, 
                value=0.6, 
                step=0.05,
                help="Minimum confidence required for identification. Values below this will be marked as 'Unknown Author'."
            )
            
            # 添加文件上传选项
            st.subheader("Or Upload Text File")
            uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "html"])
        
        # 分析按钮
        analyze_button = st.button("Analyze Text", type="primary", use_container_width=True)
        
        # 结果容器
        result_container = st.container()
        
        # 处理上传的文件
        if uploaded_file is not None and not text_input:
            text_input = uploaded_file.getvalue().decode("utf-8")
            
        # 分析逻辑
        if analyze_button:
            if not text_input.strip():
                st.error("Please enter text to analyze or upload a file!")
            else:
                # 显示处理状态
                with st.spinner("Analyzing..."):
                    # 使用会话中的识别器实例
                    result = st.session_state.identifier.analyze_text(
                        text_input, 
                        confidence_threshold=confidence_threshold
                    )
                
                # 显示结果
                with result_container:
                    st.subheader("Analysis Results:")
                    
                    # 显示主要结果
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        author = result['predicted_author']
                        st.info(f"**Predicted Author**: {author}")
                    with col2:
                        st.info(f"**Confidence**: {result['confidence']:.2f}")
                    with col3:
                        if 'num_chunks_analyzed' in result:
                            st.info(f"**Text Chunks Analyzed**: {result['num_chunks_analyzed']}")
                    
                    # 如果有多个块，显示作者分布
                    if 'author_distribution' in result and len(result['author_distribution']) > 1:
                        st.subheader("Author Distribution in Text Chunks:")
                        
                        # 将分布数据转换为表格格式
                        dist_data = {"Author": [], "Chunks": [], "Percentage": []}
                        total_chunks = result['num_chunks_analyzed']
                        
                        for author, count in result['author_distribution'].items():
                            dist_data["Author"].append(author)
                            dist_data["Chunks"].append(count)
                            dist_data["Percentage"].append(f"{count/total_chunks*100:.1f}%")
                        
                        st.dataframe(dist_data)
                    
                    # 显示所有分类概率
                    st.subheader("All Category Probabilities:")
                    
                    # 按概率从高到低排序
                    sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                    
                    # 创建表格以显示所有概率
                    prob_data = {"Author": [], "Probability": []}
                    for author, prob in sorted_probs:
                        prob_data["Author"].append(author)
                        prob_data["Probability"].append(f"{prob:.4f}")
                    
                    st.dataframe(prob_data)
    
    with tab2:
        # 风格生成器标签页
        st.title("Author Style Text Generator")
        st.markdown("This tool can generate text in famous authors' styles using two different models for comparison.")
        
        # 在会话状态中初始化Style API（如果不存在）
        if 'style_api' not in st.session_state:
            # 使用Hugging Face的个人令牌（在生产环境中应保持安全）
            token = "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"
            st.session_state.style_api = StreamlitAuthorStyleAPI(token)
        
        # 作家选择
        api = st.session_state.style_api
        author = st.selectbox(
            "Select author style to generate:",
            api.available_authors,
            help="Choose the writing style of the author you want to mimic."
        )
        
        # 生成设置
        with st.expander("Generation Settings", expanded=False):
            col1, col2 = st.columns(2)
            
            with col1:
                max_length = st.slider(
                    "Maximum Text Length", 
                    min_value=50, 
                    max_value=500, 
                    value=200, 
                    step=10,
                    help="Maximum length of generated text (in tokens)."
                )
            
            with col2:
                num_samples = st.slider(
                    "Number of Samples", 
                    min_value=1, 
                    max_value=10, 
                    value=3, 
                    step=1,
                    help="How many samples to generate to find the best one."
                )
        
        # 生成按钮
        generate_button = st.button("Generate Text", type="primary", use_container_width=True)
        
        # 结果容器
        generation_container = st.container()
        
        # 生成逻辑
        if generate_button:
            with generation_container:
                st.subheader("Generated Text Results:")
                
                # 为模型输出创建标签页
                model_tab1, model_tab2, model_tab3 = st.tabs(["Our Model", "DeepSeek Model", "Comparison"])
                
                try:
                    # 使用自定义模型生成
                    with model_tab1:
                        with st.spinner(f"Generating text in {author}'s style..."):
                            best_sample, score = api.generate_best_sample(author, num_samples, max_length)
                            
                            st.markdown("### Text Generated by Our Custom Model")
                            st.markdown(f"*Style Match Score: {score:.4f}*")
                            st.markdown("---")
                            st.markdown(best_sample)
                    
                    # 存储到会话状态以供比较
                    st.session_state.best_sample = best_sample
                    st.session_state.score = score
                    
                    # 用DeepSeek生成
                    with model_tab2:
                        deepseek_text = streamlit_chat_with_deepseek(author)
                        
                        # 评估DeepSeek文本风格匹配分数
                        deepseek_score = api.evaluate_text(deepseek_text, author)
                        
                        st.markdown("### Text Generated by DeepSeek Model")
                        st.markdown(f"*Style Match Score: {deepseek_score:.4f}*")
                        st.markdown("---")
                        st.markdown(deepseek_text)
                    
                    # 存储到会话状态以供比较
                    st.session_state.deepseek_text = deepseek_text
                    st.session_state.deepseek_score = deepseek_score
                    
                    # 比较结果
                    with model_tab3:
                        st.markdown("### Model Comparison")
                        
                        # 创建比较表
                        comparison_data = {
                            "Model": ["Our Custom Model", "DeepSeek LLM"],
                            "Style Match Score": [f"{score:.4f}", f"{deepseek_score:.4f}"]
                        }
                        
                        st.dataframe(comparison_data)
                        
                        # 确定哪个模型表现更好
                        if score > deepseek_score:
                            st.success("Our custom model generated text with a higher style match score.")
                        elif deepseek_score > score:
                            st.success("The DeepSeek model generated text with a higher style match score.")
                        else:
                            st.info("Both models generated text with the same style match score.")
                        
                        # 显示并排比较
                        st.markdown("### Side-by-Side Text Comparison")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Our Model:**")
                            st.text_area("", best_sample, height=400, disabled=True)
                        
                        with col2:
                            st.markdown("**DeepSeek Model:**")
                            st.text_area("", deepseek_text, height=400, disabled=True)
                
                except Exception as e:
                    st.error(f"Error during text generation: {str(e)}")
    
    with tab3:
        # 关于页面
        st.title("About the Author Style Tool")
        st.markdown("""
        ### Project Introduction
        
        This application provides two main features:
        
        1. **Author Style Identifier**: A natural language processing tool that analyzes text and identifies style characteristics to infer potential author styles.
        
        2. **Author Style Generator**: A text generation tool that creates new text in the style of famous authors using specialized language models.
        
        ### Technical Implementation
        
        This project uses the following technologies:
        - Pre-trained language models (BERT for identification, specialized models for generation)
        - Custom fine-tuned models for author style generation
        - DeepSeek API integration for comparison generation
        - PyTorch deep learning framework
        - Streamlit web interface
        
        ### Model Capabilities
        
        The application supports analysis and generation for the following authors:
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
        
        ### Usage Instructions
        
        #### Text Analysis:
        1. Go to the "Style Analysis" tab
        2. Enter or upload the text to be analyzed
        3. Adjust the confidence threshold as needed
        4. Click "Analyze Text" to see the results
        
        #### Text Generation:
        1. Go to the "Style Generation" tab
        2. Select an author style from the dropdown list
        3. Adjust generation settings as needed
        4. Click "Generate Text" to create samples
        5. Compare outputs from different models
        """)
        
        # 显示更多模型信息
        if 'model_info' in st.session_state:
            model_info = st.session_state.model_info
            st.subheader("Detailed Model Information")
            st.json(model_info)

if __name__ == "__main__":
    main()
