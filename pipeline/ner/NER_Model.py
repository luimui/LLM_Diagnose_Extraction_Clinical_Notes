import torch 
from IPython.display import HTML, display
import nbimporter

from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    BertModel
)
import sys
import os


class NER_Model:

    def __init__(self):

    
        #NER Model
        self.HF_TOKEN = ''
        self.DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        self.ner_label_list = ["O", "B-PROBLEM", "I-PROBLEM"]
        self.NER_NUM_LABELS = len(self.ner_label_list)
        self.NER_LABEL_TO_ID = {label: idx for idx, label in enumerate(self.ner_label_list)}
        self.NER_ID_TO_LABEL = {idx: label for idx, label in enumerate(self.ner_label_list)}
    
    
    def load_ner_model_from_directory(self
                                      , NER_MODEL_PATH=""
                                      , NER_MODEL_NAME=""
                                      , HF_TOKEN=None
                                      , NER_NUM_LABELS=None
                                      , NER_LABEL_TO_ID=None
                                      , NER_ID_TO_LABEL=None
                                      , DEVICE=None):
 
        print(f"Loading NER model from directory: {NER_MODEL_PATH}")

        if HF_TOKEN is None:
            HF_TOKEN=self.HF_TOKEN
        if NER_NUM_LABELS is None:
            NER_NUM_LABELS=self.NER_NUM_LABELS
        if NER_LABEL_TO_ID is None:
            NER_LABEL_TO_ID=self.NER_LABEL_TO_ID
        if NER_ID_TO_LABEL is None:
            NER_ID_TO_LABEL=self.NER_ID_TO_LABEL
        if DEVICE is None:
            DEVICE=self.DEVICE
    
    
        try:
           
            # Tokenizer laden
            tokenizer = AutoTokenizer.from_pretrained(NER_MODEL_NAME, token=self.HF_TOKEN)
            #tokenizer.to(DEVICE)
            
            # Load model (automatically detects safetensors)
            model = AutoModelForTokenClassification.from_pretrained(pretrained_model_name_or_path=NER_MODEL_PATH
                                                                    , num_labels=self.NER_NUM_LABELS
                                                                    , id2label=self.NER_ID_TO_LABEL
                                                                    , label2id=self.NER_LABEL_TO_ID
                                                                    , ignore_mismatched_sizes=True )
            
            
            # Move the model to the appropriate device
            model.to(self.DEVICE)
            
            print("✅ Model and tokenizer loaded successfully!")
            return model, tokenizer
            
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return None, None
    
    
    
    
    def ner_disease_terms(self, text, model, tokenizer, split="window"):# Tokenizer laden
    
        tokens_whole_doc = []
        predicted_labels_whole_dock = []
    
        
        if split == "window":
            # tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
            # model.resize_token_embeddings(len(tokenizer))
            inputs = tokenizer(    text,
                                    return_overflowing_tokens=True,
                                    truncation=True,
                                    max_length=64,
                                    padding='max_length',
                                    stride=0,  # overlap of 64 tokens
                                    return_tensors='pt',  # optional: return PyTorch tensors
                                    
                                )
    
            input_ids_chunks = inputs['input_ids']
            attention_mask_chunks = inputs['attention_mask']
    
            # Loop through each chunk
            for input_ids, attention_mask in zip(input_ids_chunks, attention_mask_chunks):
                # Move tensors to the correct device (CPU/GPU)
                input_ids = input_ids.unsqueeze(0).to(self.DEVICE)  # Add batch dimension
                attention_mask = attention_mask.unsqueeze(0).to(self.DEVICE)
            
                # Run the model for prediction
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get logits (raw model outputs)
                logits = outputs.logits
        
                # Vorhersagen ermitteln
                predictions = torch.argmax(logits, dim=2)
                
                # Vorhergesagte Labels abrufen        
                predicted_labels = [self.NER_ID_TO_LABEL[pred.item()] for pred in predictions[0]]
                
                # Spezialtokens entfernen ([CLS] und [SEP])
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                tokens = tokens[1:-1]
                predicted_labels = predicted_labels[1:-1]
        
                tokens_whole_doc.extend(tokens)
                predicted_labels_whole_dock.extend(predicted_labels)
    
    
    
    
        else:
            raise ValueError("Missing split method for text, bigger than 512 tokens") 
        
    
    
          
        return tokens_whole_doc, predicted_labels_whole_dock




    def ner_disease_terms_for_color_printing(self, text, model, tokenizer, split="window"):# Tokenizer laden
    
        tokens_whole_doc = []
        predicted_labels_whole_dock = []
    
        
        if split == "window":
            tokenizer.add_special_tokens({"additional_special_tokens": ["\n"]})
            model.resize_token_embeddings(len(tokenizer))
            inputs = tokenizer(    text,
                                    return_overflowing_tokens=True,
                                    truncation=True,
                                    max_length=64,
                                    padding='max_length',
                                    stride=0,  # overlap of 64 tokens
                                    return_tensors='pt',  # optional: return PyTorch tensors
                                    
                                )
    
            input_ids_chunks = inputs['input_ids']
            attention_mask_chunks = inputs['attention_mask']
    
            # Loop through each chunk
            for input_ids, attention_mask in zip(input_ids_chunks, attention_mask_chunks):
                # Move tensors to the correct device (CPU/GPU)
                input_ids = input_ids.unsqueeze(0).to(self.DEVICE)  # Add batch dimension
                attention_mask = attention_mask.unsqueeze(0).to(self.DEVICE)
            
                # Run the model for prediction
                with torch.no_grad():
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                
                # Get logits (raw model outputs)
                logits = outputs.logits
        
                # Vorhersagen ermitteln
                predictions = torch.argmax(logits, dim=2)
                
                # Vorhergesagte Labels abrufen        
                predicted_labels = [self.NER_ID_TO_LABEL[pred.item()] for pred in predictions[0]]
                
                # Spezialtokens entfernen ([CLS] und [SEP])
                tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
                tokens = tokens[1:-1]
                predicted_labels = predicted_labels[1:-1]
        
                tokens_whole_doc.extend(tokens)
                predicted_labels_whole_dock.extend(predicted_labels)
    
    
    
    
        else:
            raise ValueError("Missing split method for text, bigger than 512 tokens") 
        
    
    
          
        return tokens_whole_doc, predicted_labels_whole_dock
    
    
    def display_colored(self, tokens, labels):
        def color_token(token, label):
            color = {
                "B-PROBLEM": "#ff3333",
                "I-PROBLEM": "#ff8a33",
                "O": "black"
            }.get(label, "gray")  # default to gray if unknown
            return f'<span style="color: {color}; font-weight: bold; margin-right: 4px;">{token}</span>'
        
        html = " ".join(color_token(t, l) for t, l in zip(tokens, labels))
        display(HTML(html))
    
    
    
    
    def readable_ner(self, model, tokenizer, text):
    
        tokens_whole_doc, predicted_labels_whole_dock = ner_disease_terms(text, model, tokenizer)    
        
        display_colored(tokens_whole_doc, predicted_labels_whole_dock)
    
        return tokens_whole_doc, predicted_labels_whole_dock




