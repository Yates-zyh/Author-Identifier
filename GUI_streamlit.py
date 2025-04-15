"""
使用以下命令启动本地 Web 应用程序：
streamlit run GUI_streamlit.py
启动 Streamlit 后，请在浏览器中访问：http://127.0.0.1:8501
"""
import streamlit as st
from identify import AuthorIdentifier
from generate_with_chatbot import AuthorStyleAPI, chat_with_deepseek, compare_style
import os
import dotenv
import logging
import json

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("streamlit_app")

# 加载环境变量
dotenv.load_dotenv()

def author_style_analysis(tab):
    """文本作者风格分析功能"""
    # 页面标题
    tab.title("Author Style Identifier")
    tab.markdown("This tool can analyze text and identify potential author styles.")
    
    # 检查环境变量中是否存在 IDENTIFICATION_TOKEN
    token_from_env = os.environ.get("IDENTIFICATION_TOKEN")
    
    # 如果环境变量中没有 token，显示 token 输入框
    if not token_from_env:
        tab.warning("No Hugging Face token found in environment variables. The model will be loaded from public repository.")
        tab.info("Optional: You can provide your Hugging Face token for better model access.")
        
        # Token 输入框
        user_token = tab.text_input(
            "Enter your Hugging Face token (optional):",
            type="password",
            help="Your token will be used for this session only and won't be stored.",
            key="id_token"
        )
    else:
        user_token = None  # 如果环境变量中有 token，则使用环境变量中的 token
        tab.success("Hugging Face token found in environment variables.")
    
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
        uploaded_file = st.file_uploader("Choose a file", type=["txt", "md", "html"], key="id_file")
    
    # 分析按钮
    analyze_button = tab.button("Analyze Text", type="primary", use_container_width=True, key="id_analyze")
    
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
            with st.spinner("Initializing model..."):
                # 根据是否有用户提供的 token 初始化 identifier
                token_to_use = user_token if user_token else token_from_env
                identifier = AuthorIdentifier(token=token_to_use)
                
                # 获取模型信息并存储到会话状态
                model_info = identifier.get_model_info()
                st.session_state.model_info = model_info
            
            with st.spinner("Analyzing..."):
                # 分析文本
                result = identifier.analyze_text(
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
                
                # 显示模型来源信息
                tab.subheader("Model Information:")
                tab.info(f"Model Path: {model_info.get('model_path', 'Unknown')}")
                tab.info(f"Mode: {model_info.get('mode', 'Unknown')}")
                tab.info(f"Device: {model_info.get('device', 'Unknown')}")

def text_generation(tab):
    """文本生成功能，分为三个子页面"""
    # 页面标题
    tab.title("Author Style Generator")
    tab.markdown("Generate text in the style of different authors using various models.")
    
    # 创建三个子标签页
    gen_tab1, gen_tab2, gen_tab3 = tab.tabs([
        "Local Model Generation", 
        "DeepSeek Generation",
        "Style Comparison"
    ])
    
    # 调用各子页面的功能
    with gen_tab1:
        local_model_generation(gen_tab1)
    
    with gen_tab2:
        deepseek_generation(gen_tab2)
    
    with gen_tab3:
        style_comparison(gen_tab3)

def local_model_generation(tab):
    """使用本地模型生成文本功能"""
    tab.subheader("Generate Text with Local Model")
    tab.markdown("Generate text in the style of different authors using our own fine-tuned language models.")
    
    # 检查环境变量中是否存在 GENERATION_TOKEN
    token_from_env = os.environ.get("GENERATION_TOKEN")
    
    # 如果环境变量中没有 token，显示 token 输入框
    if not token_from_env:
        tab.warning("No Hugging Face token found in environment variables. The model will be loaded from public repository.")
        tab.info("Optional: You can provide your Hugging Face token for better model access.")
        
        # Token 输入框
        user_token = tab.text_input(
            "Enter your Hugging Face token (optional):",
            type="password",
            help="Your token will be used for this session only and won't be stored.",
            key="local_gen_token"
        )
    else:
        user_token = None  # 如果环境变量中有 token，则使用环境变量中的 token
        tab.success("Hugging Face token found in environment variables.")
    
    # 初始化 token
    token_to_use = user_token if user_token else token_from_env
    
    # 作者选择和输入区域
    col1, col2 = tab.columns([3, 1])
    
    # 所有可用作者
    available_authors = [
        "Agatha_Christie",
        "Alexandre_Dumas",
        "Arthur_Conan_Doyle",
        "Charles_Dickens",
        "Charlotte_Brontë",
        "F._Scott_Fitzgerald",
        "García_Márquez",
        "Herman_Melville",
        "Jane_Austen",
        "Mark_Twain"
    ]
    
    with col1:
        # 提示输入区域
        prompt_input = tab.text_area(
            "Enter prompt for text generation (optional):", 
            height=150, 
            help="Enter a prompt to start the generated text. Leave empty for open-ended generation."
        )
    
    with col2:
        # 选择作者
        selected_author = tab.selectbox(
            "Select Author Style:",
            available_authors,
            help="Choose the author whose style you want to generate text in.",
            key="local_author"
        )
        
        # 生成设置
        tab.subheader("Generation Settings")
        
        num_samples = tab.slider(
            "Number of Samples", 
            min_value=1, 
            max_value=3, 
            value=1, 
            step=1,
            help="Number of different text samples to generate.",
            key="local_samples"
        )
        
        max_length = tab.slider(
            "Maximum Length", 
            min_value=50, 
            max_value=500, 
            value=200, 
            step=50,
            help="Maximum length of the generated text.",
            key="local_length"
        )
    
    # 生成按钮
    generate_button = tab.button(
        "Generate Text", 
        type="primary", 
        use_container_width=True,
        key="local_generate_btn"
    )
    
    # 结果容器
    result_container = tab.container()
    
    # 使用本地模型生成文本
    if generate_button:
        with result_container:
            with st.spinner(f"Generating text in the style of {selected_author}..."):
                try:
                    # 初始化 API 并生成文本
                    api = AuthorStyleAPI(token=token_to_use)
                    
                    # 将生成的文本和作者保存到会话状态中，以便在比较页面使用
                    samples = api.generate_text(
                        selected_author,
                        prompt=prompt_input,
                        num_samples=num_samples,
                        max_length=max_length
                    )
                    
                    # 保存到会话状态
                    if 'local_generated_text' not in st.session_state:
                        st.session_state.local_generated_text = {}
                    
                    st.session_state.local_generated_text = {
                        'author': selected_author,
                        'prompt': prompt_input,
                        'samples': samples
                    }
                    
                    # 显示生成的文本
                    tab.subheader(f"Generated Text in the Style of {selected_author}")
                    
                    for i, sample in enumerate(samples):
                        tab.markdown(f"**Sample {i+1}:**")
                        tab.markdown(f"```\n{sample}\n```")
                        tab.markdown("---")
                except Exception as e:
                    tab.error(f"Error generating text: {str(e)}")

def deepseek_generation(tab):
    """使用 DeepSeek 生成文本功能"""
    tab.subheader("Generate Text with DeepSeek API")
    tab.markdown("Generate text in the style of different authors using DeepSeek's powerful language model.")
    
    # 检查环境变量中是否存在 OPENAI_API_KEY
    deepseek_api_key = os.environ.get("OPENAI_API_KEY")
    
    # API 密钥处理
    if not deepseek_api_key:
        tab.warning("No DeepSeek API key found in environment variables.")
        tab.info("Please provide your DeepSeek API key to enable text generation.")
        
        # DeepSeek API 密钥输入
        user_api_key = tab.text_input(
            "Enter DeepSeek API Key:",
            type="password",
            help="Your DeepSeek API key will be used for this session only.",
            key="deepseek_key_input"
        )
        
        if not user_api_key:
            tab.error("DeepSeek API key is required for text generation.")
            deepseek_enabled = False
        else:
            deepseek_api_key = user_api_key
            deepseek_enabled = True
            tab.success("DeepSeek API key provided successfully.")
    else:
        deepseek_enabled = True
        tab.success("DeepSeek API key found in environment variables.")
    
    # 作者选择和输入区域
    col1, col2 = tab.columns([3, 1])
    
    # 所有可用作者
    available_authors = [
        "Agatha_Christie",
        "Alexandre_Dumas",
        "Arthur_Conan_Doyle",
        "Charles_Dickens",
        "Charlotte_Brontë",
        "F._Scott_Fitzgerald",
        "García_Márquez",
        "Herman_Melville",
        "Jane_Austen",
        "Mark_Twain"
    ]
    
    with col1:
        # 提示输入区域
        prompt_input = tab.text_area(
            "Enter prompt for text generation (optional):", 
            height=150, 
            help="Enter a prompt to start the generated text. Leave empty for open-ended generation.",
            key="deepseek_prompt_input"  # 添加唯一的key
        )
    
    with col2:
        # 选择作者
        selected_author = tab.selectbox(
            "Select Author Style:",
            available_authors,
            help="Choose the author whose style you want to generate text in.",
            key="deepseek_author"
        )
    
    # 生成按钮（根据是否有API密钥决定是否禁用）
    generate_button = tab.button(
        "Generate with DeepSeek", 
        type="primary", 
        use_container_width=True,
        disabled=not deepseek_enabled,
        key="deepseek_generate_btn"
    )
    
    # 结果容器
    result_container = tab.container()
    
    # 使用 DeepSeek API 生成文本
    if generate_button and deepseek_enabled:
        with result_container:
            with st.spinner(f"Generating text with DeepSeek API in the style of {selected_author}..."):
                try:
                    # 使用 DeepSeek API 生成文本
                    deepseek_text = chat_with_deepseek(
                        selected_author,
                        prompt=prompt_input,
                        api_key=deepseek_api_key
                    )
                    
                    # 保存到会话状态，以便在比较页面使用
                    if 'deepseek_generated_text' not in st.session_state:
                        st.session_state.deepseek_generated_text = {}
                    
                    st.session_state.deepseek_generated_text = {
                        'author': selected_author,
                        'prompt': prompt_input,
                        'text': deepseek_text
                    }
                    
                    # 显示生成的文本
                    tab.subheader(f"DeepSeek Generated Text in the Style of {selected_author}")
                    tab.markdown(f"```\n{deepseek_text}\n```")
                except Exception as e:
                    tab.error(f"Error generating text with DeepSeek: {str(e)}")

def style_comparison(tab):
    """比较两种生成方式的文本风格"""
    tab.subheader("Compare Generated Text Styles")
    tab.markdown("Compare text generated by local model and DeepSeek to see which better matches the author's style.")
    
    # 检查是否有已生成的文本
    has_local_text = 'local_generated_text' in st.session_state and st.session_state.local_generated_text
    has_deepseek_text = 'deepseek_generated_text' in st.session_state and st.session_state.deepseek_generated_text
    
    if not (has_local_text or has_deepseek_text):
        tab.info("Generate text using both Local Model and DeepSeek first to compare their styles.")
        return
    
    # 显示已生成的文本
    col1, col2 = tab.columns(2)
    
    with col1:
        tab.subheader("Local Model Generated Text")
        if has_local_text:
            local_data = st.session_state.local_generated_text
            tab.markdown(f"**Author**: {local_data['author']}")
            if local_data['prompt']:
                tab.markdown(f"**Prompt**: {local_data['prompt']}")
            
            # 如果有多个样本，只显示第一个用于比较
            if local_data['samples']:
                tab.markdown(f"```\n{local_data['samples'][0]}\n```")
        else:
            tab.info("No text generated with Local Model yet.")
    
    with col2:
        tab.subheader("DeepSeek Generated Text")
        if has_deepseek_text:
            deepseek_data = st.session_state.deepseek_generated_text
            tab.markdown(f"**Author**: {deepseek_data['author']}")
            if deepseek_data['prompt']:
                tab.markdown(f"**Prompt**: {deepseek_data['prompt']}")
            
            tab.markdown(f"```\n{deepseek_data['text']}\n```")
        else:
            tab.info("No text generated with DeepSeek yet.")
    
    # 如果两种方式都有生成文本，并且是同一作者，则可以进行比较
    if has_local_text and has_deepseek_text:
        local_author = st.session_state.local_generated_text['author']
        deepseek_author = st.session_state.deepseek_generated_text['author']
        
        if local_author == deepseek_author:
            # 比较按钮
            compare_button = tab.button(
                "Compare Writing Styles", 
                type="primary", 
                use_container_width=True,
                key="compare_styles_btn"
            )
            
            # 评估结果容器
            evaluation_container = tab.container()
            
            if compare_button:
                with evaluation_container:
                    with st.spinner("Evaluating text style matching..."):
                        try:
                            token = os.environ.get("GENERATION_TOKEN")
                            
                            # 获取要比较的文本
                            local_text = st.session_state.local_generated_text['samples'][0]
                            deepseek_text = st.session_state.deepseek_generated_text['text']
                            author = local_author
                            
                            # 显示加载状态
                            status_message = st.info("Loading discriminator model...")
                            
                            try:
                                result = compare_style(author, local_text, deepseek_text, token)
                                
                                # 清除状态消息
                                status_message.empty()
                                
                                # 显示评估结果
                                tab.subheader("Style Matching Evaluation")
                                
                                score_col1, score_col2 = tab.columns(2)
                                with score_col1:
                                    tab.info(f"Local Model Style Match Score: {result['local_score']:.4f}")
                                
                                with score_col2:
                                    tab.info(f"DeepSeek Style Match Score: {result['deepseek_score']:.4f}")
                                
                                # 比较结果
                                tab.subheader("Comparison Result")
                                if result['better_match'] == 'local':
                                    tab.success(f"📊 Local Model generated text better matches {author}'s style (score: {result['local_score']:.4f} vs {result['deepseek_score']:.4f}).")
                                elif result['better_match'] == 'deepseek':
                                    tab.success(f"🌟 DeepSeek generated text better matches {author}'s style (score: {result['deepseek_score']:.4f} vs {result['local_score']:.4f}).")
                                else:
                                    tab.info("⚖️ Both models generated text with identical style matching scores.")
                                
                                # 添加评分解释
                                tab.markdown("""
                                **Score interpretation**:
                                - Higher scores indicate better style matching with the selected author
                                - Scores range from 0 (not matching) to 1 (perfect match)
                                - Scores above 0.7 typically indicate strong style resemblance
                                """)
                                
                            except Exception as e:
                                # 清除状态消息并显示错误
                                status_message.empty()
                                tab.error(f"Error loading discriminator model: {str(e)}")
                                tab.warning("Unable to use discriminator model. Please try again or check your Hugging Face token.")
                        except Exception as e:
                            tab.error(f"Error comparing text styles: {str(e)}")
        else:
            tab.warning("Cannot compare styles for different authors. Please generate text for the same author with both models.")

def about_page(tab):
    """关于页面"""
    # 页面标题
    tab.title("About the Author Style Tool")
    tab.markdown("""
    ### Project Introduction
    
    This application provides a natural language processing tool that analyzes text, identifies potential author styles, and generates text in the style of different authors.
    
    ### Technical Implementation
    
    This project uses:
    - Pre-trained BERT language model for author identification
    - Fine-tuned GPT-2 models for author-style text generation
    - DeepSeek API for advanced text generation capabilities
    - PyTorch deep learning framework
    - Streamlit web interface
    - Hugging Face model hub for model hosting
    
    ### Model Information
    
    The models used in this application:
    - Author identification model: `Yates-zyh/author_identifier`
    - Text generation models: `fjxddy/author-stylegan`
    
    ### Supported Authors
    
    The current version supports the following authors:
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
    
    1. **Style Analysis Tab**: Upload text to identify its author style
    2. **Text Generation Tab**: Generate new text in the style of selected authors
    3. **About Tab**: Learn more about the project and available models
    
    ### API Token Configuration
    
    For optimal performance, you can configure the following API tokens in your .env file:
    
    - `IDENTIFICATION_TOKEN`: For accessing Hugging Face author identification models
    - `GENERATION_TOKEN`: For accessing Hugging Face text generation models
    - `OPENAI_API_KEY`: For accessing DeepSeek API (with base URL set to DeepSeek)
    
    If no tokens are configured, the app will attempt to use publicly available models.
    """)
    
    # 显示环境配置状态
    tab.subheader("Environment Configuration Status")
    
    token_status = {
        "IDENTIFICATION_TOKEN": "✅ Configured" if os.environ.get("IDENTIFICATION_TOKEN") else "❌ Not configured",
        "GENERATION_TOKEN": "✅ Configured" if os.environ.get("GENERATION_TOKEN") else "❌ Not configured",
        "OPENAI_API_KEY": "✅ Configured" if os.environ.get("OPENAI_API_KEY") else "❌ Not configured"
    }
    
    tab.dataframe({"Token": list(token_status.keys()), "Status": list(token_status.values())})
    
    # 显示更多模型信息
    if 'model_info' in st.session_state:
        model_info = st.session_state.model_info
        tab.subheader("Detailed Model Information")
        tab.json(model_info)

def main():
    """主函数，应用程序入口点"""
    # 设置页面标题和配置
    st.set_page_config(page_title="Author Style Tool", layout="wide")
    
    # 创建包含分析工具、生成工具和关于信息的标签页
    tab1, tab2, tab3 = st.tabs([
        "Style Analysis", 
        "Text Generation",
        "About"
    ])
    
    # 调用各标签页的功能
    with tab1:
        author_style_analysis(tab1)
    
    with tab2:
        text_generation(tab2)
    
    with tab3:
        about_page(tab3)

if __name__ == "__main__":
    main()
