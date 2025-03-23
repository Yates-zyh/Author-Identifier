from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载保存的模型和分词器
model_path = "author_style_model"
tokenizer = BertTokenizer.from_pretrained(model_path)
model = BertForSequenceClassification.from_pretrained(model_path)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def analyze_text_style(text):
    """分析文本是否符合作者的写作风格"""
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
    author_style_prob = probabilities[0][1].item()
    
    return {
        "author_style_probability": author_style_prob,
        "is_author_style": author_style_prob > 0.5,
        "confidence": max(author_style_prob, 1-author_style_prob)
    }

# 测试示例
test_texts = [
    "A police inspector had come forward with a very young medical student who was completing his forensic training at the municipal dispensary, and it was they who had ventilated the room and covered the body while waiting for Dr. Urbino to arrive. They greeted him with a solemnity that on this occasion had more of condolence than veneration, for no one was unaware of the degree of his friendship with Jeremiah de Saint-Amour. The eminent teacher shook hands with each of them, as he always did with every one of his pupils before beginning the daily class in general clinical medicine, and then, as if it were a flower, he grasped the hem of the blanket with the tips of his index finger and his thumb, and slowly uncovered the body with sacramental circumspection. Jeremiah de Saint-Amour was completely naked, stiff and twisted, eyes open, body blue, looking fifty years older than he had the night before. He had luminous pupils, yellowish beard and hair, and an old scar sewn with baling knots across his stomach. The use of crutches had made his torso and arms as broad as a galley slave’s, but his defenseless legs looked like an orphan’s. Dr. Juvenal Urbino studied him for a moment, his heart aching as it rarely had in the long years of his futile struggle against death.",
    "I confess that when first I made acquaintance with Charles Strickland I never for a moment discerned that there was in him anything out of the ordinary. Yet now few will be found to deny his greatness. I do not speak of that greatness which is achieved by the fortunate politician or the successful soldier; that is a quality which belongs to the place he occupies rather than to the man; and a change of circumstances reduces it to very discreet proportions. The Prime Minister out of office is seen, too often, to have been but a pompous rhetorician, and the General without at! army is but the tame hero of a market town. The greatness of Charles Strickland was authentic. It may be that you do not like his art, but at all events you can hardly refuse it the tribute of your interest. He disturbs and arrests. The time has passed when he was an object of ridicule, and it is no longer a mark of eccentricity to defend or of perversity to extol him. His faults are accepted as the necessary complement to his merits. It is still possible to discuss his place in art, and the adulation of his admirers is perhaps no less capricious than the .disparagement of his detractors; but one thing cantilever be doubtful, and that is that he had genius. Tjo rqfy mind the most interesting thing in art is the personality of the artist ; and if that is singu- lar, I am willing to excuse a thousand faults. I suppose Velasquez was a better painter than El Greco, but custom stales one’s admiration for him: the Cretan, sensual and tragic, proffers the mystery of his soul like a standing sacrifice. The artist, painter, poet, or musician, by his decoration, sublime or beautiful, satisfies the aesthetic sense; but that is akin to the sexual instinct, anjl shares its barbarity: he lays before you also the greater gift of himself. To pursue his secret has something of the fascination of a detective story. It is a riddle which shares with the universe the merit of having no answer. The most insignificant of Strickland’s works suggests a personality which is strange, tormented, and complex; and it is this surely which prevents even those who do not like his pictures from being indifferent to them; it is this which has excited so curious an interest in his life and character. "
]

for text in test_texts:
    result = analyze_text_style(text)
    print(f"文本: {text[:50]}...")
    print(f"作者风格概率: {result['author_style_probability']:.2f}")
    print(f"判断结果: {'符合作者风格' if result['is_author_style'] else '不符合作者风格'}")
    print(f"置信度: {result['confidence']:.2f}")
    print("-" * 50)
