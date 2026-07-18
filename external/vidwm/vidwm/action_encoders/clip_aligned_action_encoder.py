import numpy as np
import torch
import torch.nn as nn
import einops
import datetime
import os
from tqdm.auto import tqdm

from vidwm.action_encoders.base_action_encoder import ActionEncoderBase
   
            
class ActionEncoderCLIPAligned(ActionEncoderBase):
    """
    Action Encoder aligned with semantic space of CLIP
    """
    def __init__(self, action_dim, action_num: int = 7, hidden_dim: int = 1024, text_cond: bool = True):
        super().__init__()
        
        # initialize parameters
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_dim = hidden_dim
        self.text_cond = text_cond
        
        # create model
        input_dim = self.action_dim
        
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )
        
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.action_encode[2].weight, mode='fan_in', nonlinearity='relu')

    def transform_action_to_text(self, actions: torch.tensor, precision: int = 3):
        """
        Transform num
        
        :param self: Description
        :param actions: Description
        :type actions: torch.tensor
        :param precision: Description
        :type precision: int
        """
        # action: (B, action_num, action_dim)
        B, T, D = actions.shape
        
        # output text actions
        text_actions = actions.float().cpu().numpy() # .reshape(B * T, D)
        
        # action format
        action_format = "translate x: %.3f, y: %.3f, z: %.3f; rotate x: %.3f, y: %.3f, z: %.3f; gripper position: %.3f"
        
        # transform to text
        text_actions = np.array([
            [action_format % tuple(frame_act) for frame_act in text_act]
            for text_act in text_actions
        ])
            
        # alternative
        # text_actions = [np.char.mod(action_format, tuple(text_act)) for text_act in text_actions]
                       
        # # create iterator
        # text_it = iter(text_actions)
        
        # # reshape to (B, T, 1)
        # text_actions = [
        #     [[next(text_it)] for _ in range(T)]
        #     for _ in range(B)
        # ]
        
        return text_actions

    def forward(
        self, 
        action, 
        texts=None, 
        text_tokenizer=None, 
        text_encoder=None, 
        frame_level_cond=True, 
        precision: int = 3,
        text_encoder_is_vit: bool = False,
        device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    ):
        # action: (B, action_num, action_dim)
        B, T, D = action.shape
        
        if not frame_level_cond:
            action = einops.rearrange(action, 'b t d -> b 1 (t d)')
            
        # compute action embeddings
        action_embeds = self.action_encode(action)
        
        # transform actions to text
        text_actions = self.transform_action_to_text(actions=action, precision=precision)
        
        # init
        action_with_text_embeds = action_embeds
        text_embeds = None

        # encode text actions and instruction
        with torch.no_grad():
            # embeddings from transformed actions
            text_action_embeds = self.encode_text(
                texts=text_actions.reshape(-1).tolist(),
                text_tokenizer=text_tokenizer,
                text_encoder=text_encoder,
                text_encoder_is_vit=text_encoder_is_vit,
                device=device,
            )
            
            # reshape the text-action embeds
            text_action_embeds = text_action_embeds.reshape(B, T, text_action_embeds.shape[-1])
            
            # encode text instruction
            if texts is not None and self.text_cond:
                # text instruction
                text_embeds = self.encode_text(
                    texts=texts,
                    text_tokenizer=text_tokenizer,
                    text_encoder=text_encoder,
                    text_encoder_is_vit=text_encoder_is_vit,
                    device=device,
                )
                
                # reshape
                if len(text_embeds.shape) == 2:
                    text_embeds = einops.rearrange(text_embeds, 'b c -> b 1 c') # (B, 1, 1024)
            
                # combine action with text
                action_with_text_embeds = action_embeds + text_embeds # (B, T, hidden_dim)
            
        # output (B, 1, hidden_dim) or (B, T, hidden_dim) if frame_level_cond
        output = {
            "action_with_text_embeds": action_with_text_embeds,
            "action_embeds": action_embeds,
            "text_embeds": text_embeds,
            "text_action_embeds": text_action_embeds,
        }
            
        return output 
