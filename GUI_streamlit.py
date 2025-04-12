"""
使用以下命令启动本地 Web 应用程序：
streamlit run Identify_streamlit.py
启动 Streamlit 后，请在浏览器中访问：http://127.0.0.1:8501
"""
import streamlit as st
from identify import AuthorIdentifier
import os

def main():
    # 设置页面标题和配置
    st.set_page_config(page_title="Author Style Identifier", layout="wide")
    
    # 创建包含分析工具和关于信息的标签页
    tab1, tab2 = st.tabs(["Analysis Tool", "About"])
    
    with tab1:
        # 页面标题
        st.title("Author Style Identifier")
        st.markdown("This tool can analyze text and identify the potential author's style.")
        
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
                help="Paste the text to analyze here."
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
                help="Minimum confidence required for identification. Below this value will be marked as 'Unknown Author'."
            )
            
            # 添加文件上传选项
            st.subheader("Or Upload Text File")
            uploaded_file = st.file_uploader("Select File", type=["txt", "md", "html"])
        
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
                        # 将"未知作家"转换为"Unknown Author"
                        author = result['predicted_author']
                        if author == "未知作家":
                            author = "Unknown Author"
                        st.info(f"**Predicted Author**: {author}")
                    with col2:
                        st.info(f"**Confidence**: {result['confidence']:.2f}")
                    with col3:
                        if 'num_chunks_analyzed' in result:
                            st.info(f"**Text Chunks Analyzed**: {result['num_chunks_analyzed']}")
                    
                    # 如果有多个块，显示作者分布
                    if 'author_distribution' in result and len(result['author_distribution']) > 1:
                        st.subheader("Author Distribution Across Text Chunks:")
                        
                        # 将分布数据转换为表格格式
                        dist_data = {"Author": [], "Chunk Count": [], "Percentage": []}
                        total_chunks = result['num_chunks_analyzed']
                        
                        for author, count in result['author_distribution'].items():
                            # 将"未知作家"转换为"Unknown Author"
                            if author == "未知作家":
                                author = "Unknown Author"
                            dist_data["Author"].append(author)
                            dist_data["Chunk Count"].append(count)
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
        # 关于页面
        st.title("About Author Style Identifier")
        st.markdown("""
        ### Project Introduction
        
        The Author Style Identifier is a natural language processing tool that can analyze text and identify style characteristics to infer the potential author's style.
        
        ### Technical Implementation
        
        This project uses the following technologies:
        - Pre-trained language model (BERT)
        - PyTorch deep learning framework
        - Streamlit web interface
        
        ### Usage Instructions
        
        1. Enter or upload text to analyze in the "Analysis Tool" tab
        2. Adjust the confidence threshold
        3. Click the "Analyze Text" button
        4. View the analysis results
        
        ### Data Sources
        
        Training data includes works from many famous authors, such as:
        - Jane Austen
        - Charles Dickens
        - Arthur Conan Doyle
        - And more
        """)
        
        # 显示更多模型信息
        if 'model_info' in st.session_state:
            model_info = st.session_state.model_info
            st.subheader("Detailed Model Information")
            st.json(model_info)

if __name__ == "__main__":
    main()
