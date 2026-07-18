import numpy as np
import torch
import torch.nn as nn
import einops
import datetime
import os
from tqdm.auto import tqdm



class ActionEncoderBase(nn.Module):
    def __init__(self):
        super().__init__()

    def encode_text(
        self, 
        texts, 
        text_tokenizer, 
        text_encoder, 
        text_encoder_is_vit: bool = False,
        output_dim: int = 1024,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        """
        Encode text using tokenizer and encoder
        
        :param self: Description
        :param texts: Description
        :param text_tokenizer: Description
        :param text_encoder: Description
        :param text_encoder_is_vit: Description
        :type text_encoder_is_vit: bool
        """
        if text_encoder_is_vit:
            # tokenize
            inputs = text_tokenizer(texts, padding='max_length', return_tensors="pt", truncation=True).to(device)
            
            # encode
            outputs = text_encoder(**inputs)
            text_embeds = outputs.text_embeds # (B, 512)
            
            if output_dim == 1024:
                # repeat to map from dim=512 to dim=1024
                text_embeds = einops.repeat(text_embeds, 'b c -> b 1 (n c)', n=2) # (B, 1, 1024)
        else:
            # tokenize
            text_tokens = text_tokenizer(texts).to(device)
            
            # encode
            text_embeds = text_encoder.encode_text(text_tokens) # (B, 1024)
            
        return text_embeds
            
 