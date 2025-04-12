import json
import os
import torch
from transformers import BertTokenizer, BertForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple, Union, Optional
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("author_identifier")

class AuthorIdentifier:
    """
    Author Identifier
    Used to predict the style of text authors
    """
    
    def __init__(self, model_path: str = None, device: str = None):
        """
        Initialize the Author Identifier
        
        Parameters:
        - model_path: Path to the model, default is None, will try to auto-detect the latest model
        - device: Device to use, default is None (auto-detection)
        """
        self.model_path = "../author_style_model"
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Loading model: {self.model_path}")
        logger.info(f"Using device: {self.device}")
        
        # Load tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained(self.model_path)
        self.model = BertForSequenceClassification.from_pretrained(self.model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Load labels
        try:
            with open(os.path.join(self.model_path, "label_names.json"), 'r', encoding='utf-8') as f:
                self.label_names = json.load(f)
            logger.info(f"Loaded {len(self.label_names)} labels: {', '.join(self.label_names)}")
        except FileNotFoundError:
            logger.warning(f"Label file not found: {os.path.join(self.model_path, 'label_names.json')}")
            self.label_names = [f"Category{i}" for i in range(self.model.config.num_labels)]
            
        # Load model metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self) -> Dict:
        """Load model metadata"""
        metadata_path = os.path.join(self.model_path, "model_metadata.json")
        if os.path.exists(metadata_path):
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                logger.info(f"Model metadata loaded")
                return metadata
            except Exception as e:
                logger.warning(f"Failed to load model metadata: {str(e)}")
        
        logger.warning(f"Metadata file not found: {metadata_path}")
        return {}
    
    def _batch_texts(self, text: str, max_length: int = 512, overlap: int = 128, 
                     min_chunk_length: int = 100) -> List[str]:
        """
        Split long text into multiple overlapping batches for processing
        
        Parameters:
        - text: Input text
        - max_length: Maximum token length
        - overlap: Number of overlapping tokens
        - min_chunk_length: Minimum chunk length (tokens)
        
        Returns:
        - chunks: List of text batches
        """
        # Ensure text is cleaned, removing extra whitespaces
        text = ' '.join(text.split())
        
        # Tokenize
        tokens = self.tokenizer.encode(text)
        
        chunks = []
        start_idx = 0
        
        # Use sliding window to split text
        while start_idx < len(tokens):
            # Ensure not exceeding text length
            end_idx = min(start_idx + max_length, len(tokens))
            
            window_tokens = tokens[start_idx:end_idx]
            
            # Only add chunks that are long enough
            if len(window_tokens) >= min_chunk_length:
                window_text = self.tokenizer.decode(window_tokens)
                chunks.append(window_text)
            
            # Exit loop if reaching the end of text
            if end_idx == len(tokens):
                break
                
            # Update the start position of the next window (consider overlap)
            start_idx += (max_length - overlap)
        
        return chunks
    
    def analyze_text(self, text: str, confidence_threshold: float = 0.7, 
                     return_all_chunks: bool = False) -> Dict:
        """
        Analyze the style of text authors
        
        Parameters:
        - text: Input text
        - confidence_threshold: Confidence threshold, below which "Unknown Author" will be returned
        - return_all_chunks: Whether to return analysis results for all text chunks
        
        Returns:
        - result: Dictionary containing prediction results
        """
        # If the text is too long, split into multiple chunks for analysis
        if len(text.split()) > 200:  # Approximately more than 200 words, split into chunks
            chunks = self._batch_texts(text)
            logger.info(f"Text divided into {len(chunks)} chunks for analysis")
        else:
            chunks = [text]
            
        all_chunk_results = []
        all_probabilities = []
        
        # Process each text chunk
        for i, chunk in enumerate(chunks):
            if len(chunks) > 1:
                logger.info(f"Processing chunk {i+1}/{len(chunks)}...")
                
            chunk_result = self._analyze_single_chunk(chunk, confidence_threshold)
            all_chunk_results.append(chunk_result)
            all_probabilities.append(chunk_result["raw_probabilities"])
        
        # Aggregate probabilities from multiple chunks
        if len(all_probabilities) > 1:
            # Combine probabilities from all chunks (using average)
            avg_probabilities = np.mean(all_probabilities, axis=0)
            
            # Get the highest probability and its corresponding category
            max_prob_idx = np.argmax(avg_probabilities)
            max_prob = avg_probabilities[max_prob_idx]
            
            # Create final result based on aggregated probabilities
            if max_prob < confidence_threshold:
                final_author = "Unknown Author"
                final_confidence = 1 - max_prob  # Use uncertainty as confidence
            else:
                final_author = self.label_names[max_prob_idx]
                final_confidence = max_prob
                
            # Generate final result
            final_result = {
                "predicted_author": final_author,
                "confidence": float(final_confidence),
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, avg_probabilities)},
                "num_chunks_analyzed": len(chunks)
            }
            
            # Count predictions for each chunk
            author_counts = {}
            for res in all_chunk_results:
                author = res["predicted_author"]
                if author not in author_counts:
                    author_counts[author] = 0
                author_counts[author] += 1
                
            final_result["author_distribution"] = author_counts
            
            # If needed, add results for each chunk
            if return_all_chunks:
                final_result["chunk_results"] = all_chunk_results
        else:
            # Only one chunk, directly use its result
            final_result = all_chunk_results[0]
            final_result["num_chunks_analyzed"] = 1
            
        return final_result
    
    def _analyze_single_chunk(self, text: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Analyze the style of a single text chunk
        
        Parameters:
        - text: Input text
        - confidence_threshold: Confidence threshold
        
        Returns:
        - result: Dictionary containing prediction results
        """
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=512
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=1)
        
        # Get the highest probability and its corresponding category
        max_prob, predicted_class = torch.max(probabilities, dim=1)
        max_prob = max_prob.item()
        predicted_class = predicted_class.item()
        
        # Save raw probabilities for further processing
        raw_probabilities = probabilities[0].cpu().numpy()
        
        if max_prob < confidence_threshold:
            result = {
                "predicted_author": "Unknown Author",
                "confidence": 1 - max_prob,  # Use uncertainty as confidence
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, raw_probabilities)},
                "raw_probabilities": raw_probabilities
            }
        else:
            result = {
                "predicted_author": self.label_names[predicted_class],
                "confidence": max_prob,
                "probabilities": {name: float(prob) for name, prob in zip(self.label_names, raw_probabilities)},
                "raw_probabilities": raw_probabilities
            }
        
        return result
    
    def analyze_file(self, file_path: str, confidence_threshold: float = 0.7) -> Dict:
        """
        Analyze the style of text authors in a file
        
        Parameters:
        - file_path: Path to the file
        - confidence_threshold: Confidence threshold
        
        Returns:
        - result: Dictionary containing prediction results
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                text = f.read()
            
            result = self.analyze_text(text, confidence_threshold)
            result["file_path"] = file_path
            return result
        except Exception as e:
            logger.error(f"Failed to analyze file: {str(e)}")
            return {
                "error": f"Failed to analyze file: {str(e)}",
                "file_path": file_path
            }
    
    def get_model_info(self) -> Dict:
        """
        Get model information
        
        Returns:
        - info: Dictionary containing model information
        """
        info = {
            "model_path": self.model_path,
            "device": str(self.device),
            "labels": self.label_names,
            "num_labels": len(self.label_names),
        }
        
        # Add metadata information
        info.update(self.metadata)
        
        return info

# Simplified API function
def analyze_text(text: str, confidence_threshold: float = 0.7) -> Dict:
    """
    Simplified API function for analyzing text author style
    
    Parameters:
    - text: Input text
    - confidence_threshold: Confidence threshold
    
    Returns:
    - result: Dictionary containing prediction results
    """
    identifier = AuthorIdentifier()
    return identifier.analyze_text(text, confidence_threshold)