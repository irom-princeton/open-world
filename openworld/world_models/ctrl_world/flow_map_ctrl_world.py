# from diffusers import StableVideoDiffusionPipeline
from .pipeline_stable_video_diffusion import StableVideoDiffusionPipeline
from .pipeline_flow_map_ctrl_world import CtrlWorldDiffusionPipeline
from .flow_map_unet_spatio_temporal_condition import UNetSpatioTemporalConditionModel
from .flow_map_utils import create_targets_shortcut, create_targets, create_targets_flow_matching, create_targets_lsd, create_targets_psd, create_targets_one_step

import numpy as np
import torch
import torch.nn as nn
import einops
from accelerate import Accelerator
import datetime
import os
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import json
from decord import VideoReader, cpu
import wandb
import swanlab
import mediapy
import math

import functools
import torch.utils.checkpoint as ckpt
from torch.func import jvp
from torch.nn.attention import sdpa_kernel, SDPBackend
from diffusers.models.embeddings import Timesteps, TimestepEmbedding

# _orig = ckpt.checkpoint
# ckpt.checkpoint = functools.partial(_orig, use_reentrant=False)

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb

class Action_encoder2(nn.Module):
    def __init__(self, action_dim, action_num, hidden_size, text_cond=True):
        super().__init__()
        self.action_dim = action_dim
        self.action_num = action_num
        self.hidden_size = hidden_size
        self.text_cond = text_cond

        input_dim = int(action_dim)
        self.action_encode = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024),
            nn.SiLU(),
            nn.Linear(1024, 1024)
        )
        # kaiming initialization
        nn.init.kaiming_normal_(self.action_encode[0].weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_normal_(self.action_encode[2].weight, mode='fan_in', nonlinearity='relu')

    def forward(self, action,  texts=None, text_tokinizer=None, text_encoder=None, frame_level_cond=True,):
        # action: (B, action_num, action_dim)
        B,T,D = action.shape
        if not frame_level_cond:
            action = einops.rearrange(action, 'b t d -> b 1 (t d)')
        action = self.action_encode(action)

        if texts is not None and self.text_cond:
            # with 50% probability, add text condition
            with torch.no_grad():
                inputs = text_tokinizer(texts, padding='max_length', return_tensors="pt", truncation=True).to(text_encoder.device)
                outputs = text_encoder(**inputs)
                hidden_text = outputs.text_embeds # (B, 512)
                hidden_text = einops.repeat(hidden_text, 'b c -> b 1 (n c)', n=2) # (B, 1, 1024)
            
            action = action + hidden_text # (B, T, hidden_size)
        return action # (B, 1, hidden_size) or (B, T, hidden_size) if frame_level_cond


class TimeLogvarWeight(nn.Module):
    """
    Learn a per-sample log-variance / log-weight from time information.

    forward(t, dt_base) returns a (B,) tensor of log-weights.
    - t:       (B,) in [0,1]
    - dt_base: (B,) or None. If provided, we approximate s = t - dt_base.
    """

    def __init__(self, base_channels: int, time_embed_dim: int, scale: float = 1000.0):
        super().__init__()
        self.scale = scale

        # time projection and embedding for s and t
        self.time_proj_s = Timesteps(
            base_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )
        self.time_proj_t = Timesteps(
            base_channels,
            flip_sin_to_cos=True,
            downscale_freq_shift=0,
        )

        self.time_embedding_s = TimestepEmbedding(base_channels, time_embed_dim)
        self.time_embedding_t = TimestepEmbedding(base_channels, time_embed_dim)

        # small MLP head → scalar logvar per sample
        self.head = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, 1),
        )
        # start near zero so weighting initially ~unweighted
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, t: torch.Tensor, dt_base: torch.Tensor | None = None) -> torch.Tensor:
        """
        Returns logvar: (B,)
        """
        device = t.device
        dtype = t.dtype
        t = t.view(-1)

        if dt_base is not None:
            dt_base = dt_base.to(device=device, dtype=dtype).view(-1)
            s = (t - dt_base).clamp(0.0, 1.0)
        else:
            s = torch.zeros_like(t, device=device, dtype=dtype)

        # map [0,1] → "timestep domain" used by Timesteps
        ts_s = (s * self.scale).to(dtype)
        ts_t = (t * self.scale).to(dtype)

        # (B, base_channels)
        emb_s_proj = self.time_proj_s(ts_s)
        emb_t_proj = self.time_proj_t(ts_t)

        # (B, time_embed_dim)
        emb_s = self.time_embedding_s(emb_s_proj)
        emb_t = self.time_embedding_t(emb_t_proj)

        # combine s,t embeddings; average is simple & symmetric
        emb = 0.5 * (emb_s + emb_t)  # (B, time_embed_dim)

        logvar = self.head(emb).view(-1)  # (B,)
        return logvar


class CrtlWorld(nn.Module):
    def __init__(self, args):
        super(CrtlWorld, self).__init__()

        self.args = args

        # distance conditioning for flow-map unet
        self.distance_conditioning = args.distance_conditioning
        
        self.flow_map_type = args.flow_map_type
        self.flow_map_loss_type = args.flow_map_loss_type
        
        # for flowmap
        self.psd_sample_mode = args.psd_sample_mode 
        
        # learned logvar for flow matching loss weighting
        self.use_weights = args.use_weights
        
        # for shortcut targets
        self.DENOISE_TIMESTEPS = args.DENOISE_TIMESTEPS
        
        # sigma min/max from EDM
        self.SIGMA_MIN = args.SIGMA_MIN
        self.SIGMA_MAX = args.SIGMA_MAX
        self.single_bs_mode = args.single_bs_mode
        self.bootstrap_bs = args.bootstrap_bs
        
        # probability of using one-step target in PSD training
        self.one_step_prob = args.one_step_prob
        self.one_step_sample = args.one_step_sample
        self.bias_prob = args.bias_prob
        
        print("Bias probability for PSD sampling:", self.bias_prob)
        
        # load from pretrained stable video diffusion
        # self.pipeline = StableVideoDiffusionPipeline.from_pretrained(args.svd_model_path)
        self.pipeline = CtrlWorldDiffusionPipeline.from_pretrained(args.svd_model_path)
        # repalce the unet to support frame_level pose condition
        print("replace the unet to support action condition and frame_level pose!")
        unet = UNetSpatioTemporalConditionModel(distance_conditioning=self.distance_conditioning)
        unet.load_state_dict(self.pipeline.unet.state_dict(), strict=False)
        
        # from diffusers.models.attention_processor import AttnProcessor2_0
        # unet.set_attn_processor(AttnProcessor2_0())
        
        self.pipeline.unet = unet
        
        self.unet = unet
        
        # print("Attn processor:", type(self.unet.attn_processors[list(self.unet.attn_processors.keys())[0]]))
        
        self.vae = self.pipeline.vae
        self.image_encoder = self.pipeline.image_encoder
        self.scheduler = self.pipeline.scheduler

        # freeze vae, image_encoder, enable unet gradient ckpt
        self.vae.requires_grad_(False)
        self.image_encoder.requires_grad_(False)
        self.unet.requires_grad_(True)
        if self.flow_map_type == 'flow_map' and self.flow_map_loss_type == 'flow_matching':
            print("enable unet gradient checkpointing!")
            self.unet.enable_gradient_checkpointing()
        
        # nonreentrant_ckpt = functools.partial(ckpt.checkpoint, use_reentrant=False)
        # # Patch the UNet and all submodules that store the checkpoint func (diffusers pattern)
        # for m in self.unet.modules():
        #     if hasattr(m, "_gradient_checkpointing_func"):
        #         m._gradient_checkpointing_func = nonreentrant_ckpt

        # SVD is a img2video model, load a clip text encoder
        from transformers import AutoTokenizer, CLIPTextModelWithProjection
        self.text_encoder = CLIPTextModelWithProjection.from_pretrained(args.clip_model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(args.clip_model_path,use_fast=False)
        self.text_encoder.requires_grad_(False)

        # initialize an action projector
        self.action_encoder = Action_encoder2(action_dim=args.action_dim, action_num=int(args.num_history+args.num_frames), hidden_size=1024, text_cond=args.text_cond)
        
        # initialize the weight module for flow-matching loss
        if self.use_weights:
            # TODO: tune the parameters
            self.log_var_net = TimeLogvarWeight(
                base_channels=128,
                time_embed_dim=256,
                scale=1000.0,
            )
        
    
    
    def _predict_v(self, x_t_full, t, dt_base, action_hidden, added_time_ids, condition_latent, num_history): 
        """
        convert EDM diffusion to flow-matching v-prediction
        x_t_full: (B, F, C, H, W)  where F = num_history + num_frames
                history part should be whatever you want to feed as condition (we'll pass noisy_history)
                future part is the shortcut x_t (interpolated between x0 and x1)
        t:       (B,) float in [0,1]
        dt_base: (B,) int (like in shortcut code), OPTIONAL but recommended
        """
        # breakpoint()
        device = self.unet.device
        B = x_t_full.shape[0]
        
        # TODO: can move noisy_history creation here for clarity
        noisy_history = x_t_full[:, :num_history] # this is in EDM space already, where noisy_history = c_in_h * (history + sigma_h * noise)
        x_t = x_t_full.clone()
        
        # reshape for broadcasting
        t_ = t.to(device).float().view(B, 1, 1, 1, 1).clamp(1/(1+self.SIGMA_MAX), 1/(1+self.SIGMA_MIN))
        sigma = ((1.0 - t_) / t_).clamp(min=self.SIGMA_MIN, max=self.SIGMA_MAX)    # (B,1,1,1,1)

        c_in   = 1.0 / torch.sqrt(sigma**2 + 1.0)       # (B,1,1,1,1)
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out  = -sigma / torch.sqrt(sigma**2 + 1.0)
        c_noise = (sigma.log() / 4.0).view(B)           # (B,)

        # map flow point to EDM point: x_edm = x_t / t = x1 + sigma*x0
        x_t_edm = x_t / t_

        # input_latents = torch.cat([noisy_history, c_in*noisy_latents[:,num_history:]], dim=1) # (B, num_history+num_frames, 4, 32, 32)
        input_latents = torch.cat([noisy_history, c_in * x_t_edm[:, num_history:]], dim=1)  # (B, F, C, H, W)
        input_latents = torch.cat([input_latents, condition_latent/self.vae.config.scaling_factor], dim=2)
        
        if dt_base is not None and self.flow_map_type == 'shortcut' and self.distance_conditioning:
            log2T = float(int(math.log2(self.DENOISE_TIMESTEPS)))
            distance = (dt_base.to(device).float() / log2T).clamp(0.0, 1.0)  # (B,)
        elif self.distance_conditioning and self.flow_map_type == 'flow_map':
            distance = dt_base
        else:
            distance = None
        
        # prediction from edm
        model_pred = self.unet(input_latents, c_noise, distance=distance, encoder_hidden_states=action_hidden, added_time_ids=added_time_ids,frame_level_cond=self.args.frame_level_cond).sample
        
        # now convert this to v-prediction in flow-map space
        x1_hat = c_out * model_pred + c_skip * x_t_edm 
        
        predicted_noise = (x_t_edm - x1_hat) / sigma
        v_pred = x1_hat - predicted_noise
        
        return v_pred[:, num_history:]  # (B, Ff, C, H, W)
    
    
    def _flow_map(self, s, t, Is, 
         action_hidden, added_time_ids, condition_latent, num_history):

        dt_base = t-s
        
        phi_st = self._predict_v(
            Is, s, dt_base,
            action_hidden=action_hidden,
            added_time_ids=added_time_ids,
            condition_latent=condition_latent,
            num_history=num_history,
        )
        
        X_st = Is[:, num_history:] + phi_st * (t - s).view(-1, 1, 1, 1, 1)
        return X_st



    def _model_partial_t(self, s, t, Is, 
         action_hidden, added_time_ids, condition_latent, num_history):
        """
        the partial derivative of model output w.r.t. time t, based on flow map formulation
        """
        
        # t_ = t.detach().clone().requires_grad_(True)
        
        def f(t_in: torch.Tensor):
            # output: (B, F_future, C, H, W)
            return self._flow_map(
                s, t_in, Is,
                action_hidden=action_hidden,
                added_time_ids=added_time_ids,
                condition_latent=condition_latent,
                num_history=num_history,
            )

        v = torch.ones_like(t)
        
        # from torch.func import jvp
        # v_pred, dv_dt = jvp(f, (t_,), (v,))
        
        # Xst, dt_Xst = torch.autograd.functional.jvp(
        #     f, (t,), (v,),
        #     create_graph=False,
        #     strict=True,
        # )
        
        # Xst, dt_Xst = jvp(f, (t,), (v,))
        with sdpa_kernel(SDPBackend.MATH):
            Xst, dt_Xst = jvp(f, (t,), (v,))
        
        return Xst, dt_Xst
        
    
    
    # ====== flow-matching shortcut targets ======
    def forward(self, batch):
        #breakpoint()
        latents = batch['latent']  # (B, F, 4, 32, 32)
        texts = batch['text']
        device = self.unet.device
        dtype = self.unet.dtype
        noise_aug_strength = 0.0

        num_history = self.args.num_history
        latents = latents.to(device) #[B, num_history + num_frames]
        bsz, num_frames = latents.shape[:2]

        # current img as condition image to stack at channel wise, add random noise to current image, noise strength 0.0~0.2
        current_img = latents[:, num_history:(num_history+1)]  # (B,1,4,32,32)
        current_img = current_img[:, 0]                        # (B,4,32,32)
        sigma = torch.rand([bsz, 1, 1, 1], device=device) * 0.2
        c_in = 1 / (sigma**2 + 1) ** 0.5
        current_img = c_in * (current_img + torch.randn_like(current_img) * sigma)
        condition_latent = einops.repeat(current_img, 'b c h w -> b f c h w', f=num_frames)
        if self.args.his_cond_zero:
            condition_latent[:, :num_history] = 0.0

        # action condition
        action = batch['action'].to(device)  # (B,f,7)
        action_hidden = self.action_encoder(action, texts, self.tokenizer, self.text_encoder, frame_level_cond=self.args.frame_level_cond) # (B, f, 1024)

        #  for classifier-free guidance, with 5% probability, set action_hidden to 0
        uncond_hidden_states = torch.zeros_like(action_hidden)
        text_mask = (torch.rand(action_hidden.shape[0], device=device) > 0.05).unsqueeze(1).unsqueeze(2)
        action_hidden = action_hidden * text_mask + uncond_hidden_states * (~text_mask)

        ##################################### Flow Matching + Shortcut ####################################
        
        # add 0~0.3 noise to history, history as condition
        history = latents[:, :num_history] # (B, num_history, 4, 32, 32)
        sigma_h = torch.randn([bsz, num_history, 1, 1, 1], device=device) * 0.3
        c_in_h = 1 / (sigma_h**2 + 1) ** 0.5
        noisy_history = c_in_h * (history + sigma_h * torch.randn_like(history)) # (B, num_history, 4, 32, 32)

        # added_time_ids
        motion_bucket_id = self.args.motion_bucket_id
        fps = self.args.fps
        added_time_ids = self.pipeline._get_add_time_ids(fps, motion_bucket_id, noise_aug_strength, action_hidden.dtype, bsz, 1, False)
        added_time_ids = added_time_ids.to(device)

        # ====== shortcut targets ======
        if self.flow_map_type == 'shortcut':
            if not self.single_bs_mode:
                x_t_full, v_target_future, t, dt_base =  create_targets_shortcut(
                        latents, num_history, action_hidden, condition_latent,
                        added_time_ids, 
                        self._predict_v,
                        labels=None,
                        bootstrap_bs=self.bootstrap_bs,
                        DENOISE_TIMESTEPS=self.DENOISE_TIMESTEPS,
                    )
            else: 
                # probability of using shortcut
                p_shortcut = 1/4
                use_shortcut = (torch.rand(1).item() < p_shortcut)
                # use_shortcut = False
                use_shortcut = True

                if use_shortcut:
                    x_t_full, v_target_future, t, dt_base =  create_targets_shortcut(
                        latents, num_history, action_hidden, condition_latent,
                        added_time_ids, 
                        self._predict_v,
                        labels=None,
                        bootstrap_bs=1,
                        DENOISE_TIMESTEPS=self.DENOISE_TIMESTEPS,
                    )
                else:
                    x_t_full, v_target_future, t, dt_base =  create_targets( 
                        latents, num_history, action_hidden, condition_latent,
                        added_time_ids, 
                        self._predict_v,
                        labels=None,
                        bootstrap_bs=1,
                        DENOISE_TIMESTEPS=self.DENOISE_TIMESTEPS,
                    )
        elif self.flow_map_type == 'flow_matching':
            x_t_full, v_target_future, t, dt_base =  create_targets_flow_matching(latents, num_history)

        elif self.flow_map_type == 'flow_map' and self.flow_map_loss_type == 'lsd':
            if not self.single_bs_mode:
                x_t_full, v_target_future, t, dt_base =  create_targets_lsd( 
                    latents, num_history, action_hidden, condition_latent,
                    added_time_ids, 
                    self._model_partial_t,
                    bootstrap_bs=self.bootstrap_bs,
                )
            else:
                # probability of using lsd
                p_lsd = 1/4
                use_lsd = (torch.rand(1).item() < p_lsd)
                # use_lsd = False
                # use_lsd = True
                # breakpoint()
                if use_lsd:
                    x_t_full, v_target_future, t, dt_base =  create_targets_lsd( 
                        latents, num_history, action_hidden, condition_latent,
                        added_time_ids, 
                        self._model_partial_t,
                        bootstrap_bs=1,
                    )
                else:
                    x_t_full, v_target_future, t, dt_base =  create_targets_flow_matching(latents, num_history, return_flow_map_dt=True)
        
        elif self.flow_map_type == 'flow_map' and self.flow_map_loss_type == 'psd':
            if not self.single_bs_mode:
                if not self.one_step_sample:
                    x_t_full, v_target_future, t, dt_base =  create_targets_psd( 
                        latents, num_history, action_hidden, condition_latent,
                        added_time_ids, 
                        self._predict_v,
                        bootstrap_bs=self.bootstrap_bs,
                        psd_sample_mode=self.psd_sample_mode,
                        bias_prob=self.bias_prob,
                    )
                else:
                    p_one_step = self.one_step_prob
                    use_one_step = (torch.rand(1).item() < p_one_step)
                    if use_one_step:
                        with torch.no_grad():
                            x_t_full, v_target_future, t, dt_base =  create_targets_one_step( 
                                latents, num_history, action_hidden, condition_latent,
                                added_time_ids, 
                                self._predict_v,
                                bootstrap_bs=self.bootstrap_bs,
                                psd_sample_mode=self.psd_sample_mode,
                            )
                    else:
                        x_t_full, v_target_future, t, dt_base =  create_targets_psd( 
                            latents, num_history, action_hidden, condition_latent,
                            added_time_ids, 
                            self._predict_v,
                            bootstrap_bs=self.bootstrap_bs,
                            psd_sample_mode=self.psd_sample_mode,
                        )
                           
            else:
                # probability of using psd
                p_psd = 1/4
                use_psd = (torch.rand(1).item() < p_psd)
                # use_psd = False
                # use_psd = True

                if use_psd:
                    x_t_full, v_target_future, t, dt_base =  create_targets_psd( 
                        latents, num_history, action_hidden, condition_latent,
                        added_time_ids, 
                        self._predict_v,
                        bootstrap_bs=1,
                        psd_sample_mode=self.psd_sample_mode,
                    )
                else:
                    x_t_full, v_target_future, t, dt_base =  create_targets_flow_matching(latents, num_history, return_flow_map_dt=True)
        
        elif self.flow_map_type == 'flow_map' and self.flow_map_loss_type == 'flow_matching':
            x_t_full, v_target_future, t, dt_base =  create_targets_flow_matching(latents, num_history, return_flow_map_dt=True)
        
        # ensure history fed as noisy_history (as above)
        x_t_full[:, :num_history] = noisy_history
    
        # ====== predict & loss ======
        v_pred_future = self._predict_v(x_t_full, t, dt_base,
                                        action_hidden=action_hidden,
                                       added_time_ids=added_time_ids,
                                       condition_latent=condition_latent,
                                       num_history=num_history)
        # breakpoint()
        if self.use_weights:
            log_var = self.log_var_net(t, dt_base)  # (B,)
            weighted_loss = torch.exp(-log_var).view(-1, 1, 1, 1, 1) * ((v_pred_future - v_target_future) ** 2) + log_var.view(-1, 1, 1, 1, 1)
            loss = weighted_loss.mean()
        else:  
            loss = ((v_pred_future - v_target_future) ** 2).mean()

        return loss, torch.tensor(0.0, device=device, dtype=dtype)