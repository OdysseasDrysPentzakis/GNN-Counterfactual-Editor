import torch
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt
import numpy as np
import string
import nltk
from nltk.corpus import stopwords

# Download necessary NLTK data
nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)

class AttentionCalc:
    def __init__(self, model_name='bert-base-uncased'):
        # Initialize tokenizer and model from pretrained BERT
       
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
       
        self.model.eval()  # Set model to evaluation mode
        # Get English stop words
        self.stop_words = set(stopwords.words('english'))
 
    def get_attention(self, text):
        # Tokenize input text and convert to tensor
        inputs = self.tokenizer(text, return_tensors='pt', add_special_tokens=True)
        # Get model outputs, including attention
        
        with torch.no_grad():
            outputs = self.model(**inputs, output_attentions=True)
     
        return inputs, outputs.attentions

    def process_attention(self, inputs, attention):
        # Average attention across all layers and heads
        mean_attention = torch.mean(torch.cat(attention), dim=0)
        # Sum attention for each token
        token_attention = mean_attention.sum(dim=0)[0]
        # Get tokens from input IDs
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])
        
        words = []
        word_attention = []
        current_word = ''
        current_attention = 0

        # Iterate through tokens and their attention scores
        for token, attn in zip(tokens, token_attention):
            # Skip special tokens and punctuation
            if token in ['[CLS]', '[SEP]'] or token in string.punctuation:
                continue
            # Handle subword tokens
            if token.startswith('##'):
                current_word += token[2:]
                current_attention += attn.item()
            else:
                # Add completed word and its attention
                if current_word:
                    words.append(current_word)
                    word_attention.append(current_attention)
                # Start new word
                current_word = token
                current_attention = attn.item()

        # Add last word if exists
        if current_word:
            words.append(current_word)
            word_attention.append(current_attention)

        # Filter out stop words
        content_words = []
        content_attention = []
        for word, attention in zip(words, word_attention):
            if word.lower() not in self.stop_words:
                content_words.append(word)
                content_attention.append(attention)

        # Normalize attention for content words
        content_attention = np.array(content_attention)
        content_attention = content_attention / content_attention.sum()

        return content_words, content_attention

   
    def print_attention(self, words, word_attention):
        # Print attention score for each word
        for word, attention in zip(words, word_attention):
            print(f"{word}: {attention:.4f}")

    def analyze(self, text):
        # Main analysis method
        # Get attention from model
        inputs, attention = self.get_attention(text)
        # Process attention to get content words and their scores
        content_words, content_attention = self.process_attention(inputs, attention)
        # Print and visualize results
        self.print_attention(content_words, content_attention)
       
# Usage
if __name__ == "__main__":
    # Create an instance of AttentionVisualizer
  
    calculator = AttentionCalc()
  
    # Example sentence to analyze
    sentence = "The quick brown fox jumps over the lazy dog."
    print(f"Analyzing: '{sentence}'\n")
    
    # Perform analysis
    calculator.analyze(sentence)
