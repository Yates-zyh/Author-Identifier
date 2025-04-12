"""
本地网页使用以下命令启动：
streamlit run Identify_streamlit.py
启动Streamlit后，请在浏览器中访问: http://127.0.0.1:8501
"""
import streamlit as st
from identify import analyze_text_style
import os
import argparse

def main():
    # 设置页面标题和配置
    st.set_page_config(page_title="Author Identifier", layout="wide")
    
    # 页面
    st.title("Author Style Identifier")
    st.markdown("This tool analyzes text to identify the writing style of the author.")
    text_input = st.text_area("Enter text to analyze:", height=200, 
                            help="Paste the text you want to analyze here.")
    
    # 置信度阈值滑块
    confidence_threshold = st.slider(
        "Confidence Threshold", 
        min_value=0.0, 
        max_value=1.0, 
        value=0.5, 
        step=0.05,
        help="Minimum confidence level required for a positive identification."
    )
    
    # 分析按钮
    if st.button("Analyze", type="primary"):
        if not text_input.strip():
            st.error("Please input the text to analyze!")
        else:
            # 显示处理中状态
            with st.spinner("Analyzing..."):
                # 调用原有的分析函数
                result = analyze_text_style(text_input, confidence_threshold=confidence_threshold)
            
            # 显示结果
            st.subheader("Results:")
            result_container = st.container()
            with result_container:
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**Predicted Writer**: {result['predicted_author']}")
                with col2:
                    st.info(f"**Confidence level**: {result['confidence']:.2f}")
                
                # 显示所有分类概率
                st.subheader("All category probabilities:")
                
                # 按概率从高到低排序
                sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
                
                # 创建表格显示所有概率
                prob_data = {"Author": [], "Probability": []}
                for author, prob in sorted_probs:
                    prob_data["Author"].append(author)
                    prob_data["Probability"].append(f"{prob:.4f}")
                st.table(prob_data)

if __name__ == "__main__":
    main()
