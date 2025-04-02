import tkinter as tk
from tkinter import scrolledtext, ttk
from identify import analyze_text_style

class StyleAnalyzerApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Author Identifier")
        self.root.geometry("800x600")
        self.root.configure(bg="#f5f5f5")
        
        # 创建标题
        title_label = tk.Label(root, text="Author Identifier", font=("Times New Roman", 18, "bold"), bg="#f5f5f5")
        title_label.pack(pady=10)
        
        # 创建输入文本框
        input_frame = tk.Frame(root, bg="#f5f5f5")
        input_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        input_label = tk.Label(input_frame, text="Please input the text: ", font=("Times New Roman", 12), bg="#f5f5f5")
        input_label.pack(anchor="w")
        
        self.text_input = scrolledtext.ScrolledText(input_frame, height=10, width=80, font=("Times New Roman", 11))
        self.text_input.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 创建控制面板
        control_frame = tk.Frame(root, bg="#f5f5f5")
        control_frame.pack(fill=tk.X, padx=20, pady=5)
        
        # 置信度阈值滑动条
        threshold_frame = tk.Frame(control_frame, bg="#f5f5f5")
        threshold_frame.pack(fill=tk.X, pady=5)
        
        threshold_label = tk.Label(threshold_frame, text="Confidence threshold: ", font=("Times New Roman", 11), bg="#f5f5f5")
        threshold_label.pack(side=tk.LEFT, padx=(0, 10))
        
        self.threshold_var = tk.DoubleVar(value=0.6)
        self.threshold_slider = ttk.Scale(threshold_frame, from_=0.1, to=0.9, length=300, 
                                         orient=tk.HORIZONTAL, variable=self.threshold_var)
        self.threshold_slider.pack(side=tk.LEFT)
        
        self.threshold_value_label = tk.Label(threshold_frame, text="0.60", width=4, font=("Times New Roman", 11), bg="#f5f5f5")
        self.threshold_value_label.pack(side=tk.LEFT, padx=10)
        
        self.threshold_slider.bind("<Motion>", self.update_threshold_label)
        
        # 分析按钮
        analyze_button = tk.Button(control_frame, text="Analyze", font=("Times New Roman", 12, "bold"), 
                                   bg="#4CAF50", fg="white", command=self.analyze_text)
        analyze_button.pack(pady=10)
        
        # 结果显示区域
        result_frame = tk.Frame(root, bg="#f5f5f5")
        result_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=10)
        
        result_label = tk.Label(result_frame, text="Results: ", font=("Times New Roman", 12), bg="#f5f5f5")
        result_label.pack(anchor="w")
        
        self.result_text = scrolledtext.ScrolledText(result_frame, height=10, width=80, font=("Times New Roman", 11))
        self.result_text.pack(fill=tk.BOTH, expand=True, pady=5)
        self.result_text.config(state=tk.DISABLED)
    
    def update_threshold_label(self, event=None):
        value = self.threshold_var.get()
        self.threshold_value_label.config(text=f"{value:.2f}")
    
    def analyze_text(self):
        text = self.text_input.get(1.0, tk.END).strip()
        if not text:
            self.show_result("Please input the text to analyze!")
            return
        
        threshold = self.threshold_var.get()
        result = analyze_text_style(text, confidence_threshold=threshold)
        
        # 格式化结果
        result_str = f"Predicted Writer: {result['predicted_author']}\n"
        result_str += f"Confidence level: {result['confidence']:.2f}\n\n"
        result_str += "All category probabilities: \n"
        
        # 按概率从高到低排序
        sorted_probs = sorted(result['probabilities'].items(), key=lambda x: x[1], reverse=True)
        for author, prob in sorted_probs:
            result_str += f"  - {author}: {prob:.4f}\n"
        
        self.show_result(result_str)
    
    def show_result(self, text):
        self.result_text.config(state=tk.NORMAL)
        self.result_text.delete(1.0, tk.END)
        self.result_text.insert(tk.END, text)
        self.result_text.config(state=tk.DISABLED)

if __name__ == "__main__":
    root = tk.Tk()
    app = StyleAnalyzerApp(root)
    root.mainloop()
