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
import dotenv
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("streamlit_app")

# 加载环境变量
dotenv.load_dotenv()

# 导入chatbot_with_generator中的类和函数
from generate_with_chatbot import AuthorStyleAPI, chat_with_deepseek, create_deepseek_client

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
def streamlit_chat_with_deepseek(author, api_key=None):
    """使用DeepSeek API生成指定作者风格的文本"""
    # 检查API密钥是否配置
    if not api_key:
        api_key = os.environ.get("OPENAI_API_KEY")
    
    if not api_key:
        return "Failed to generate text: No API key available."
        
    with st.spinner(f"Generating text in {author}'s style using DeepSeek..."):
        try:
            return chat_with_deepseek(author, api_key)
        except Exception as e:
            st.error(f"Error calling DeepSeek API: {str(e)}")
            return f"Failed to generate text: {str(e)}"

def test_deepseek_api(api_key):
    """测试DeepSeek API密钥是否有效"""
    try:
        client = create_deepseek_client(api_key)
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Hello!"}
            ],
            max_tokens=5,
            stream=False
        )
        return True, "API key is valid"
    except Exception as e:
        return False, str(e)

def author_style_analysis(tab):
    """文本作者风格分析功能"""
    # 页面标题
    tab.title("Author Style Identifier")
    tab.markdown("This tool can analyze text and identify potential author styles.")
    
    # 模型信息显示（仅初始化一次）
    if 'model_info' not in st.session_state:
        with st.spinner("Loading model information..."):
            try:
                identifier = AuthorIdentifier()
                st.session_state.model_info = identifier.get_model_info()
                st.session_state.identifier = identifier
            except Exception as e:
                tab.error(f"Failed to load model: {str(e)}")
                return
    
    # 显示模型信息
    with tab.expander("Model Information", expanded=False):
        model_info = st.session_state.model_info
        tab.markdown(f"**Model Path**: {model_info.get('model_path', 'Unknown')}")
        tab.markdown(f"**Device**: {model_info.get('device', 'Unknown')}")
        tab.markdown(f"**Supported Authors**: {', '.join(model_info.get('labels', ['Unknown']))}")
        tab.markdown(f"**Training Date**: {model_info.get('training_date', 'Unknown')}")
    
    # 文本输入区域
    col1, col2 = tab.columns([3, 1])
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
    analyze_button = tab.button("Analyze Text", type="primary", use_container_width=True)
    
    # 结果容器
    result_container = tab.container()
    
    # 处理上传的文件
    if uploaded_file is not None and not text_input:
        text_input = uploaded_file.getvalue().decode("utf-8")
        
    # 分析逻辑
    if analyze_button:
        if not text_input.strip():
            tab.error("Please enter text to analyze or upload a file!")
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
                tab.subheader("Analysis Results:")
                
                # 显示主要结果
                col1, col2, col3 = tab.columns(3)
                
                with col1:
                    author = result['predicted_author']
                    tab.info(f"**Predicted Author**: {author}")
                with col2:
                    tab.info(f"**Confidence**: {result['confidence']:.2f}")
                with col3:
                    if 'num_chunks_analyzed' in result:
                        tab.info(f"**Text Chunks Analyzed**: {result['num_chunks_analyzed']}")
                
                # 如果有多个块，显示作者分布
                if 'author_distribution' in result and len(result['author_distribution']) > 1:
                    tab.subheader("Author Distribution in Text Chunks:")
                    
                    # 将分布数据转换为表格格式
                    dist_data = {"Author": [], "Chunks": [], "Percentage": []}
                    total_chunks = result['num_chunks_analyzed']
                    
                    for author, count in result['author_distribution'].items():
                        dist_data["Author"].append(author)
                        dist_data["Chunks"].append(count)
                        dist_data["Percentage"].append(f"{count/total_chunks*100:.1f}%")
                    
                    tab.dataframe(dist_data)
                
                # 显示所有分类概率
                tab.subheader("All Category Probabilities:")
                
                # 按概率从高到低排序
                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                
                # 创建表格以显示所有概率
                prob_data = {"Author": [], "Probability": []}
                for author, prob in sorted_probs:
                    prob_data["Author"].append(author)
                    prob_data["Probability"].append(f"{prob:.4f}")
                
                tab.dataframe(prob_data)

def custom_model_generation(tab):
    """使用自定义模型生成文本功能"""
    # 标题和介绍
    tab.title("Custom Model Generation")
    tab.markdown("Generate text in the style of famous authors using our fine-tuned models.")
    
    # 在会话状态中初始化Style API（如果不存在）
    if 'style_api' not in st.session_state:
        # 尝试从环境变量中获取Hugging Face的令牌
        token = os.environ.get("GENERATION_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        if not token:
            tab.warning("No Hugging Face token found in environment. Will use public model from cache if available.")
        st.session_state.style_api = StreamlitAuthorStyleAPI(token)
    
    # 作家选择
    api = st.session_state.style_api
    author = tab.selectbox(
        "Select author style to generate:",
        api.available_authors,
        help="Choose the writing style of the author you want to mimic.",
        key="custom_model_author_select"  # 添加唯一key
    )
    
    # 生成设置
    with tab.expander("Generation Settings", expanded=True):
        col1, col2 = tab.columns(2)
        
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
    generate_button = tab.button(
        "Generate Text with Custom Model", 
        type="primary", 
        use_container_width=True,
        key="custom_gen_button"
    )
    
    # 结果容器
    result_container = tab.container()
    
    # 生成逻辑
    if generate_button:
        with result_container:
            # 模型状态指示器
            model_status = tab.empty()
            
            # 检查是否有token
            token = os.environ.get("GENERATION_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
            if not token:
                model_status.warning("No Hugging Face token found. Will use public model from cache.")
            
            # 生成文本
            try:
                with st.spinner(f"Generating text in {author}'s style..."):
                    best_sample, score = api.generate_best_sample(author, num_samples, max_length)
                    
                    tab.markdown("### Text Generated by Our Custom Model")
                    tab.markdown(f"*Style Match Score: {score:.4f}*")
                    tab.markdown("---")
                    tab.text_area("Generated Text:", value=best_sample, height=300, disabled=True)
                    
                    # 存储结果供后续使用
                    st.session_state.custom_result = (best_sample, score)
                    tab.success("Text generation successful!")
            except Exception as e:
                tab.error(f"Error generating text with custom model: {str(e)}")
                tab.info("Try setting GENERATION_TOKEN or HUGGINGFACE_TOKEN in your .env file for better results.")

def deepseek_model_generation(tab):
    """使用DeepSeek模型生成文本功能"""
    # 标题和介绍
    tab.title("DeepSeek Model Generation")
    tab.markdown("Generate text in the style of famous authors using DeepSeek's powerful language model.")
    
    # 在会话状态中初始化Style API（如果不存在）
    if 'style_api' not in st.session_state:
        # 尝试从环境变量中获取Hugging Face的令牌
        token = os.environ.get("GENERATION_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
        st.session_state.style_api = StreamlitAuthorStyleAPI(token)
    
    # 作家选择
    api = st.session_state.style_api
    author = tab.selectbox(
        "Select author style to generate:",
        api.available_authors,
        help="Choose the writing style of the author you want to mimic.",
        key="deepseek_model_author_select"  # 添加唯一key
    )
    
    # API密钥设置
    tab.subheader("DeepSeek API Configuration")
    
    # 读取环境变量中的API密钥
    env_api_key = os.environ.get("OPENAI_API_KEY")
    
    # 决定是否显示API密钥
    if env_api_key:
        tab.success("DeepSeek API key found in environment variables.")
        use_env_key = tab.checkbox("Use API key from environment", value=True)
        
        if not use_env_key:
            api_key = tab.text_input(
                "Enter your DeepSeek API key:",
                type="password",
                help="Your DeepSeek API key will be used for this session only and won't be stored."
            )
        else:
            api_key = env_api_key
    else:
        tab.warning("No DeepSeek API key found in environment variables.")
        api_key = tab.text_input(
            "Enter your DeepSeek API key:",
            type="password",
            help="Your DeepSeek API key will be used for this session only and won't be stored."
        )
        
    # 测试API密钥有效性
    if api_key:
        test_button = tab.button("Test API Key", key="test_deepseek_key")
        if test_button:
            with st.spinner("Testing API key..."):
                is_valid, message = test_deepseek_api(api_key)
                if is_valid:
                    tab.success("API key is valid!")
                else:
                    tab.error(f"API key validation failed: {message}")
    
    # 生成按钮
    generate_button = tab.button(
        "Generate Text with DeepSeek", 
        type="primary", 
        use_container_width=True,
        key="deepseek_gen_button",
        disabled=not api_key
    )
    
    # 如果没有API密钥，显示提示
    if not api_key:
        tab.info("Please enter a DeepSeek API key to generate text.")
    
    # 结果容器
    result_container = tab.container()
    
    # 生成逻辑
    if generate_button and api_key:
        with result_container:
            try:
                # 生成文本
                deepseek_text = streamlit_chat_with_deepseek(author, api_key)
                
                # 评估DeepSeek文本风格匹配分数
                with st.spinner("Evaluating DeepSeek output..."):
                    deepseek_score = api.evaluate_text(deepseek_text, author)
                
                tab.markdown("### Text Generated by DeepSeek Model")
                tab.markdown(f"*Style Match Score: {deepseek_score:.4f}*")
                tab.markdown("---")
                tab.text_area("Generated Text:", value=deepseek_text, height=300, disabled=True)
                
                # 存储结果供后续使用
                st.session_state.deepseek_result = (deepseek_text, deepseek_score)
                tab.success("Text generation successful!")
            except Exception as e:
                tab.error(f"Error generating text with DeepSeek: {str(e)}")
                if "Rate limit" in str(e):
                    tab.info("You might be hitting DeepSeek's rate limits. Try again after a few minutes.")
                elif "Authentication" in str(e) or "key" in str(e).lower():
                    tab.info("There seems to be an issue with your API key. Please verify it's correct.")

def compare_model_outputs(tab):
    """比较不同模型的输出结果功能"""
    # 标题和介绍
    tab.title("Model Comparison")
    tab.markdown("Compare the output from our custom model and DeepSeek's model side by side.")
    
    # 检查是否已生成两个模型的结果
    custom_result_exists = 'custom_result' in st.session_state
    deepseek_result_exists = 'deepseek_result' in st.session_state
    
    # 提示用户需要先生成结果
    if not custom_result_exists and not deepseek_result_exists:
        tab.info("Please generate text using both models first by visiting the 'Custom Model Generation' and 'DeepSeek Model Generation' tabs.")
        return
    elif not custom_result_exists:
        tab.info("Please generate text using our custom model first by visiting the 'Custom Model Generation' tab.")
        return
    elif not deepseek_result_exists:
        tab.info("Please generate text using DeepSeek model first by visiting the 'DeepSeek Model Generation' tab.")
        return
    
    # 获取结果
    custom_text, custom_score = st.session_state.custom_result
    deepseek_text, deepseek_score = st.session_state.deepseek_result
    
    # 创建比较表
    tab.subheader("Model Performance Comparison")
    comparison_data = {
        "Model": ["Our Custom Model", "DeepSeek LLM"],
        "Style Match Score": [f"{custom_score:.4f}", f"{deepseek_score:.4f}"]
    }
    
    tab.dataframe(comparison_data)
    
    # 确定哪个模型表现更好
    if custom_score > deepseek_score:
        tab.success("Our custom model generated text with a higher style match score.")
    elif deepseek_score > custom_score:
        tab.success("The DeepSeek model generated text with a higher style match score.")
    else:
        tab.info("Both models generated text with the same style match score.")
        
    # 显示文本分析
    score_diff = abs(custom_score - deepseek_score)
    if score_diff < 0.05:
        tab.markdown("The scores are very close, suggesting both models perform similarly for this author's style.")
    elif score_diff > 0.2:
        tab.markdown("There's a significant difference in scores, indicating one model is much better at capturing this author's style.")
    
    # 显示并排比较
    tab.markdown("### Side-by-Side Text Comparison")
    col1, col2 = tab.columns(2)
    
    with col1:
        tab.markdown("**Our Custom Model:**")
        tab.text_area("", custom_text, height=400, disabled=True, key="custom_compare_area")
    
    with col2:
        tab.markdown("**DeepSeek Model:**")
        tab.text_area("", deepseek_text, height=400, disabled=True, key="deepseek_compare_area")
    
    # 添加刷新比较按钮
    if tab.button("Refresh Comparison", use_container_width=True):
        # 实际上什么都不做，只是刷新页面，获取最新的会话状态数据
        tab.rerun()

def about_page(tab):
    """关于页面"""
    # 页面标题
    tab.title("About the Author Style Tool")
    tab.markdown("""
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
    1. Go to the "Custom Model" tab to generate text with our model:
       - Select an author style
       - Adjust generation settings
       - Click "Generate Text with Custom Model"
    
    2. Go to the "DeepSeek Model" tab to generate text with DeepSeek:
       - Select an author style
       - Enter your DeepSeek API key if not configured in environment
       - Click "Generate Text with DeepSeek"
    
    3. Go to the "Model Comparison" tab to compare outputs:
       - View side-by-side comparison of both generated texts
       - Compare style matching scores
    
    ### API Token Configuration
    
    For optimal performance, you can configure the following API tokens in your .env file:
    
    - `GENERATION_TOKEN` or `HUGGINGFACE_TOKEN`: For accessing Hugging Face models
    - `OPENAI_API_KEY`: For accessing DeepSeek's API services
    - `IDENTIFICATION_TOKEN`: For accessing the author identification model
    
    If no tokens are configured, the app will attempt to use publicly available models with reduced capabilities.
    You can also enter your DeepSeek API key directly in the interface.
    """)
    
    # 显示更多模型信息
    if 'model_info' in st.session_state:
        model_info = st.session_state.model_info
        tab.subheader("Detailed Model Information")
        tab.json(model_info)
        
    # 显示环境配置状态
    tab.subheader("Environment Configuration Status")
    
    token_status = {
        "GENERATION_TOKEN": "✅ Configured" if os.environ.get("GENERATION_TOKEN") else "❌ Not configured",
        "OPENAI_API_KEY": "✅ Configured" if os.environ.get("OPENAI_API_KEY") else "❌ Not configured",
        "IDENTIFICATION_TOKEN": "✅ Configured" if os.environ.get("IDENTIFICATION_TOKEN") else "❌ Not configured",
    }
    
    tab.dataframe({"Token": list(token_status.keys()), "Status": list(token_status.values())})

def main():
    """主函数，应用程序入口点"""
    # 设置页面标题和配置
    st.set_page_config(page_title="Author Style Tool", layout="wide")
    
    # 创建包含分析工具、生成工具子标签页和关于信息的标签页
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Style Analysis", 
        "Custom Model", 
        "DeepSeek Model", 
        "Model Comparison",
        "About"
    ])
    
    # 调用各标签页的功能
    with tab1:
        author_style_analysis(tab1)
    
    with tab2:
        custom_model_generation(tab2)
    
    with tab3:
        deepseek_model_generation(tab3)
    
    with tab4:
        compare_model_outputs(tab4)
    
    with tab5:
        about_page(tab5)

if __name__ == "__main__":
    main()
