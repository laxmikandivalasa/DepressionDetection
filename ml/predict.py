"""
MindWatch - AI-Powered Mental Health Awareness Platform
Prediction and Inference Module
"""

import torch
import numpy as np
from transformers import DistilBertForSequenceClassification, DistilBertTokenizer
import os

class DepressionDetector:
    """Depression detection using fine-tuned DistilBERT"""
    
    def __init__(self, model_path='models/distilbert_depression'):
        """Initialize the detector with trained model"""
        self.model_path = model_path
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load model and tokenizer
        self.load_model()
    
    def load_model(self):
        """Load the trained model and tokenizer"""
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(
                f"Model not found at {self.model_path}. "
                "Please train the model first using ml/train_model.py"
            )
        
        print(f"Loading model from: {self.model_path}")
        self.tokenizer = DistilBertTokenizer.from_pretrained(self.model_path)
        self.model = DistilBertForSequenceClassification.from_pretrained(self.model_path,attn_implementation="eager")
        self.model.to(self.device)
        self.model.eval()
        print(f"Model loaded successfully on {self.device}")
    
    def predict(self, text, return_attention=False):
        """
        Predict depression from text
        
        Args:
            text (str): Input text to analyze
            return_attention (bool): Whether to return attention weights
        
        Returns:
            dict: Prediction results with label, confidence, and optionally attention
        """
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors='pt',
            truncation=True,
            padding=True,
            max_length=512
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=return_attention)
        
        # Get probabilities
        logits = outputs.logits
        probs = torch.softmax(logits, dim=1)
        
        # Get prediction (handle batch dimension properly)
        prediction_idx = probs.argmax(dim=1).item()
        confidence = probs.max(dim=1).values.item()
        
        # Map to labels
        label = "Depressed" if prediction_idx == 1 else "Non-Depressed"
        
        result = {
            'label': label,
            'confidence': float(confidence),
            'probabilities': {
                'Non-Depressed': float(probs[0][0].item()),
                'Depressed': float(probs[0][1].item())
            },
            'is_crisis': float(probs[0][1].item()) > 0.85  # High threshold for crisis
        }
        
        # Add attention weights if requested
        if return_attention and hasattr(outputs, "attentions") and outputs.attentions:
            # Get attention from last layer, average across heads
            attention = outputs.attentions[-1].mean(dim=1).squeeze().cpu().numpy()
            
            # Get tokens
            tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
            
            # Handle attention shape - it might be 2D (seq_len, seq_len) or 1D (seq_len,)
            if len(attention.shape) == 2:
                # Take diagonal or mean across second dimension
                attention_scores = attention.mean(axis=0)
            else:
                attention_scores = attention
            
            # Create token-attention pairs
            token_attention = [
                {'token': token, 'attention': float(attention_scores[i])}
                for i, token in enumerate(tokens)
                if token not in ['[CLS]', '[SEP]', '[PAD]']
            ]
            
            # Sort by attention weight
            token_attention = sorted(token_attention, key=lambda x: x['attention'], reverse=True)
            
            result['attention'] = token_attention[:20]  # Top 20 tokens
        
        return result
    
    def predict_batch(self, texts):
        """
        Predict depression for multiple texts
        
        Args:
            texts (list): List of texts to analyze
        
        Returns:
            list: List of prediction results
        """
        results = []
        for text in texts:
            result = self.predict(text)
            results.append(result)
        
        return results
    
    def get_emotion_breakdown(self, text):
        """
        Get detailed emotion analysis with multiple emotion categories
        
        Args:
            text (str): Input text
        
        Returns:
            dict: Comprehensive emotion breakdown
        """
        prediction = self.predict(text)
        text_lower = text.lower()
        
        # Enhanced emotion categories with more comprehensive keywords
        emotions = {
            'sadness': 0.0,
            'anxiety': 0.0,
            'anger': 0.0,
            'hopelessness': 0.0,
            'loneliness': 0.0,
            'fear': 0.0,
            'guilt': 0.0,
            'joy': 0.0,
            'contentment': 0.0,
            'excitement': 0.0
        }
        
        # Comprehensive keyword dictionaries with intensity weights
        emotion_keywords = {
            'sadness': {
                'high': ['devastated', 'heartbroken', 'miserable', 'despair', 'anguish'],
                'medium': ['sad', 'unhappy', 'depressed', 'down', 'blue', 'crying', 'tears'],
                'low': ['disappointed', 'upset', 'gloomy', 'melancholy']
            },
            'anxiety': {
                'high': ['panic', 'terrified', 'overwhelmed', 'paralyzed'],
                'medium': ['anxious', 'worried', 'nervous', 'stressed', 'tense'],
                'low': ['uneasy', 'concerned', 'restless', 'jittery']
            },
            'anger': {
                'high': ['furious', 'enraged', 'livid', 'outraged'],
                'medium': ['angry', 'mad', 'frustrated', 'irritated', 'annoyed'],
                'low': ['bothered', 'displeased', 'agitated']
            },
            'hopelessness': {
                'high': ['hopeless', 'worthless', 'suicidal', 'no point'],
                'medium': ['pointless', 'meaningless', 'give up', 'cant go on'],
                'low': ['discouraged', 'defeated', 'lost']
            },
            'loneliness': {
                'high': ['completely alone', 'abandoned', 'isolated'],
                'medium': ['lonely', 'alone', 'nobody cares', 'no one'],
                'low': ['disconnected', 'withdrawn', 'distant']
            },
            'fear': {
                'high': ['terrified', 'petrified', 'horrified'],
                'medium': ['scared', 'afraid', 'frightened', 'fearful'],
                'low': ['worried', 'apprehensive', 'nervous']
            },
            'guilt': {
                'high': ['ashamed', 'humiliated', 'mortified'],
                'medium': ['guilty', 'regret', 'sorry', 'fault'],
                'low': ['embarrassed', 'uncomfortable']
            },
            'joy': {
                'high': ['ecstatic', 'thrilled', 'overjoyed', 'elated'],
                'medium': ['happy', 'joyful', 'cheerful', 'delighted'],
                'low': ['pleased', 'glad', 'satisfied']
            },
            'contentment': {
                'high': ['peaceful', 'serene', 'blissful'],
                'medium': ['content', 'calm', 'relaxed', 'comfortable'],
                'low': ['okay', 'fine', 'alright']
            },
            'excitement': {
                'high': ['amazing', 'incredible', 'fantastic', 'wonderful'],
                'medium': ['excited', 'enthusiastic', 'eager', 'great'],
                'low': ['interested', 'curious', 'hopeful']
            }
        }
        
        # Calculate emotion scores based on keyword presence and intensity
        for emotion, intensity_dict in emotion_keywords.items():
            score = 0.0
            for intensity, keywords in intensity_dict.items():
                weight = {'high': 1.0, 'medium': 0.6, 'low': 0.3}[intensity]
                matches = sum(1 for keyword in keywords if keyword in text_lower)
                score += matches * weight
            
            # Normalize score to 0-1 range
            emotions[emotion] = min(1.0, score / 2.0)
        
        # Adjust based on depression prediction
        if prediction['label'] == 'Depressed':
            # Boost negative emotions
            for emotion in ['sadness', 'anxiety', 'hopelessness', 'loneliness']:
                emotions[emotion] = min(1.0, emotions[emotion] + prediction['confidence'] * 0.3)
            # Reduce positive emotions
            for emotion in ['joy', 'contentment', 'excitement']:
                emotions[emotion] = max(0.0, emotions[emotion] - prediction['confidence'] * 0.2)
        
        # Calculate overall mood score (0-10 scale)
        positive_score = (emotions['joy'] + emotions['contentment'] + emotions['excitement']) / 3
        negative_score = (emotions['sadness'] + emotions['anxiety'] + emotions['hopelessness'] + 
                         emotions['loneliness'] + emotions['fear'] + emotions['guilt']) / 6
        
        mood_score = (positive_score - negative_score + 1) * 5  # Convert to 0-10 scale
        mood_score = max(0, min(10, mood_score))
        
        # Determine dominant emotions (top 3)
        sorted_emotions = sorted(emotions.items(), key=lambda x: x[1], reverse=True)
        dominant_emotions = [
            {'emotion': emotion, 'intensity': round(score, 2)}
            for emotion, score in sorted_emotions[:3]
            if score > 0.1
        ]
        
        return {
            'primary_prediction': prediction,
            'emotions': emotions,
            'mood_score': round(mood_score, 1),
            'dominant_emotions': dominant_emotions,
            'sentiment': 'positive' if mood_score > 6 else 'negative' if mood_score < 4 else 'neutral'
        }


# Singleton instance for reuse
_detector_instance = None

def get_detector(model_path='models/distilbert_depression'):
    """Get or create detector instance"""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = DepressionDetector(model_path)
    return _detector_instance


if __name__ == '__main__':
    # Test the detector
    detector = DepressionDetector()
    
    # Test samples
    test_texts = [
        "I'm feeling great today! Life is wonderful.",
        "I feel so hopeless and empty. Nothing matters anymore.",
        "Just had a productive day at work. Feeling accomplished!",
        "I can't stop crying. The pain is unbearable."
    ]
    
    print("=" * 60)
    print("Testing Depression Detector")
    print("=" * 60)
    
    for text in test_texts:
        print(f"\nText: {text}")
        result = detector.predict(text, return_attention=True)
        print(f"Prediction: {result['label']}")
        print(f"Confidence: {result['confidence']:.2%}")
        print(f"Crisis Alert: {result['is_crisis']}")
        
        if 'attention' in result:
            print("Top attention tokens:", [t['token'] for t in result['attention'][:5]])
