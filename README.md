# Author Style Analyzer

A comprehensive system for author style analysis, identification, and text generation based on specific author styles.

## Overview

This project provides tools to analyze writing styles, identify authors of given texts, and generate new text in the style of specific authors. It uses state-of-the-art natural language processing techniques, including BERT for author classification and GPT-2 for style-based text generation.

## Features

- **Author Style Training**: Train models to recognize the writing styles of different authors
- **Author Identification**: Analyze text samples to identify their likely authors
- **Style-based Text Generation**: Generate new text that mimics the style of specific authors
- **User-friendly GUI**: Simple interface for analyzing texts and identifying authors

## Project Structure

- `train.py`: Script for training the author style classification model
- `identify.py`: Module for identifying authors from text samples
- `generate.py`: Module for generating text in the style of specific authors
- `Identifier_GUI.py`: Graphical user interface for the author identification system

## Requirements

- Python 3.7+
- PyTorch
- Transformers (Hugging Face)
- NLTK
- tkinter (for GUI)
- scikit-learn
- tqdm

## Installation

1. Clone the repository:
   ```
   git clone git@github.com:Yates-zyh/Author-Identifier.git
   ```

2. Install required packages:
   ```
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install -e .
   ```

3. Download required NLTK data:
   ```python
   import nltk
   nltk.download('gutenberg')
   nltk.download('brown')
   nltk.download('reuters')
   nltk.download('webtext')
   ```

## Usage

### Training the Model

Prepare your data in the following structure:
```
data/
  ├── Author1/
  │   ├── text1.txt
  │   ├── text2.txt
  │   └── ...
  ├── Author2/
  │   ├── text1.txt
  │   └── ...
  └── ...
```

Run the training script:
```
python train.py
```

### Identifying Authors

To identify the author of a text using the command line:
```python
from identify import analyze_text_style

result = analyze_text_style("Your text here", confidence_threshold=0.6)
print(f"Predicted author: {result['predicted_author']}")
print(f"Confidence: {result['confidence']:.2f}")
```

### Generating Text in an Author's Style

Generate text in the style of a specific author:
```
python generate.py --author "AuthorName" --words 200
```

### Using the GUI

Launch the graphical user interface:
```
python Identifier_GUI.py
```

1. Enter the text you want to analyze in the input box
2. Adjust the confidence threshold as needed
3. Click "Analyze" to identify the most likely author

## Model Details

The system uses:
- **BERT** (bert-base-uncased) fine-tuned for author style classification
- **GPT-2** for generating text in specific author styles

## Acknowledgments

- This project was developed as part of the EBA5004 Practical Language Processing course.
- Thanks to the Hugging Face team for their Transformers library.
