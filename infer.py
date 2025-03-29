import json
from transformers import BertTokenizer, BertForSequenceClassification
import torch

model_path = "../author_style_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def analyze_text_style(text, confidence_threshold=0.6):
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=256
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    probabilities = torch.nn.functional.softmax(logits, dim=1)
    
    # 获取最高概率及其对应的类别
    max_prob, predicted_class = torch.max(probabilities, dim=1)
    max_prob = max_prob.item()
    predicted_class = predicted_class.item()
    
    with open(f"{model_path}/label_names.json", 'r') as f:
        label_names = json.load(f)
    
    if max_prob < confidence_threshold:
        result = {
            "predicted_author": "未知作家",
            "confidence": 1 - max_prob,
            "probabilities": {name: prob.item() for name, prob in zip(label_names, probabilities[0])}
        }
    else:
        result = {
            "predicted_author": label_names[predicted_class],
            "confidence": max_prob,
            "probabilities": {name: prob.item() for name, prob in zip(label_names, probabilities[0])}
        }
    
    return result

with open("test_texts.json", "r", encoding="utf-8") as f:
    test_texts = json.load(f)

for text in test_texts:
    result = analyze_text_style(text)
    print(f"文本: {text[:50]}...")
    print(f"预测作家: {result['predicted_author']}")
    print(f"置信度: {result['confidence']:.2f}")
    print("所有类别概率:")
    for author, prob in result['probabilities'].items():
        print(f"  - {author}: {prob:.4f}")
    print("-" * 50)