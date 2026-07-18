import math
import shutil
from rich.console import Console

import json
import os
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from pathlib import Path

from typing import Any, Dict, List, Literal, Optional, Union

from typing_extensions import Annotated
import pickle
import time
import numpy as np
import mediapy as mp
import torch
from torch import nn
import einops
from tqdm import tqdm
from enum import Enum
import gc
import random
import warnings
from omegaconf import OmegaConf

from typing import Callable, Dict, List, Optional, Union
import torch
# from einops import rearrange, repeat
import PIL

from vidwm.video_models.stable_video_diffusion  import StableVideoDiffusionPipeline

from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import StableVideoDiffusionPipelineOutput
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.stable_video_diffusion.pipeline_stable_video_diffusion import _resize_with_antialiasing


from vidwm.video_models.utils.svd_model_utils import _append_dims, svd_tensor2vid


class VidWMDiffusionPipeline(StableVideoDiffusionPipeline):
    @torch.no_grad()
    def __call__(
        self,
        image,
        text,
        height: int = 576,
        width: int = 1024,
        num_frames: Optional[int] = None,
        num_inference_steps: int = 25,
        min_guidance_scale: float = 1.0,
        max_guidance_scale: float = 3.0,
        fps: int = 7,
        motion_bucket_id: int = 127,
        noise_aug_strength: int = 0.02,
        decode_chunk_size: Optional[int] = None,
        num_videos_per_prompt: Optional[int] = 1,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        callback_on_step_end: Optional[Callable[[int, int, Dict], None]] = None,
        callback_on_step_end_tensor_inputs: List[str] = ["latents"],
        return_dict: bool = True,
        mask = None,
        cond_wrist=None,
        history=None,
        frame_level_cond=False,
        his_cond_zero=False,
        enable_uq: bool = False,
        enable_action_pred: bool = False,
        enable_reward_pred: bool = False,
        flow_map_type: str = "flow_map",
        flow_map_loss_type: str = "psd",
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            image (`PIL.Image.Image` or `List[PIL.Image.Image]` or `torch.FloatTensor`):
                Image or images to guide image generation. If you provide a tensor, it needs to be compatible with
                [`CLIPImageProcessor`](https://huggingface.co/lambdalabs/sd-image-variations-diffusers/blob/main/feature_extractor/preprocessor_config.json).
            height (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to `self.unet.config.sample_size * self.vae_scale_factor`):
                The width in pixels of the generated image.
            num_frames (`int`, *optional*):
                The number of video frames to generate. Defaults to 14 for `stable-video-diffusion-img2vid` and to 25 for `stable-video-diffusion-img2vid-xt`
            num_inference_steps (`int`, *optional*, defaults to 25):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference. This parameter is modulated by `strength`.
            min_guidance_scale (`float`, *optional*, defaults to 1.0):
                The minimum guidance scale. Used for the classifier free guidance with first frame.
            max_guidance_scale (`float`, *optional*, defaults to 3.0):
                The maximum guidance scale. Used for the classifier free guidance with last frame.
            fps (`int`, *optional*, defaults to 7):
                Frames per second. The rate at which the generated images shall be exported to a video after generation.
                Note that Stable Diffusion Video's UNet was micro-conditioned on fps-1 during training.
            motion_bucket_id (`int`, *optional*, defaults to 127):
                The motion bucket ID. Used as conditioning for the generation. The higher the number the more motion will be in the video.
            noise_aug_strength (`int`, *optional*, defaults to 0.02):
                The amount of noise added to the init image, the higher it is the less the video will look like the init image. Increase it for more motion.
            decode_chunk_size (`int`, *optional*):
                The number of frames to decode at a time. The higher the chunk size, the higher the temporal consistency
                between frames, but also the higher the memory consumption. By default, the decoder will decode all frames at once
                for maximal quality. Reduce `decode_chunk_size` to reduce memory usage.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            callback_on_step_end (`Callable`, *optional*):
                A function that calls at the end of each denoising steps during the inference. The function is called
                with the following arguments: `callback_on_step_end(self: DiffusionPipeline, step: int, timestep: int,
                callback_kwargs: Dict)`. `callback_kwargs` will include a list of all tensors as specified by
                `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`List`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            enable_uq (`bool`, *optional*, defaults to `False`):
                Whether or not to predict the confidence in generated videos.
            enable_action_pred (`bool`, *optional*, defaults to `False`):
                Whether or not to predict actions along with the generated videos.
            enable_reward_pred (`bool`, *optional*, defaults to `False`):
                Whether or not to predict rewards along with the generated videos.

        Returns:
            [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`~pipelines.stable_diffusion.StableVideoDiffusionPipelineOutput`] is returned,
                otherwise a `tuple` is returned where the first element is a list of list with the generated frames.

        Examples:

        ```py
        from diffusers import StableVideoDiffusionPipeline
        from diffusers.utils import load_image, export_to_video

        pipe = StableVideoDiffusionPipeline.from_pretrained("stabilityai/stable-video-diffusion-img2vid-xt", torch_dtype=torch.float16, variant="fp16")
        pipe.to("cuda")

        image = load_image("https://lh3.googleusercontent.com/y-iFOHfLTwkuQSUegpwDdgKmOjRSTvPxat63dQLB25xkTs4lhIbRUFeNBWZzYf370g=s1200")
        image = image.resize((1024, 576))

        frames = pipe(image, num_frames=25, decode_chunk_size=8).frames[0]
        export_to_video(frames, "generated.mp4", fps=7)
        ```
        """
        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # frames
        num_frames = num_frames if num_frames is not None else self.unet.config.num_frames
        decode_chunk_size = decode_chunk_size if decode_chunk_size is not None else num_frames
        
        # device
        # device = self._execution_device
        device = self.unet.device   
        
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = max_guidance_scale > 1.0

        # # 1. Check inputs. Raise error if not correct
        # self.check_inputs(image, height, width)

        # 2. Define call parameters
        batch_size = None
        # if isinstance(image, PIL.Image.Image):
        #     batch_size = 1
        # elif isinstance(image, list):
        #     batch_size = len(image)
        # else:
        #     batch_size = image.shape[0]
        
        # 3. Encode input image or text
        # SVD uses CLIP image embeddings.
        # Here, we use CLIP text embeddings
        image_embeddings = text
        
        # input repeat factor
        inp_repeat_factor = 1
            
        # update the batch size
        batch_size = image_embeddings.shape[0]
        
        # include negative conditioning embeddings, if using classifier-free guidance
        if do_classifier_free_guidance:
            negative_image_embeddings = torch.zeros_like(image_embeddings)
            
            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            image_embeddings = torch.cat([negative_image_embeddings, image_embeddings])
        
        # # clip_image = self.video_processor.preprocess(image, height=224, width=224)
        # clip_image = _resize_with_antialiasing(image, (224, 224))
        # image_embeddings = self._encode_image(clip_image, device, num_videos_per_prompt, do_classifier_free_guidance)
        
        # NOTE: Stable Diffusion Video was conditioned on fps - 1, which
        # is why it is reduced here.
        # See: https://github.com/Stability-AI/generative-models/blob/ed0997173f98eaf8f4edf7ba5fe8f15c6b877fd3/scripts/sampling/simple_video_sample.py#L188
        # fps = fps - 1 # we only use fps = 7 in train, so just set to 7

        # 4. Encode input image using VAE
        needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast
        if image.shape[-3] == 3: # (batch, 3, 256, 256)
            image = self.video_processor.preprocess(image, height=height, width=width)
            noise = randn_tensor(image.shape, generator=generator, device=image.device, dtype=image.dtype)
            # image = image + noise_aug_strength * noise

            if needs_upcasting:
                self.vae.to(dtype=torch.float32)

            image_latents = self._encode_vae_image(
                image, 
                device=device, 
                num_videos_per_prompt=num_videos_per_prompt, 
                do_classifier_free_guidance=do_classifier_free_guidance,
            )
            image_latents = image_latents.to(image_embeddings.dtype)

            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
        else: # e.g., (batch, 4, 72, 40)
            # image latents are already computed
            image_latents = image / self.vae.config.scaling_factor
            
            if do_classifier_free_guidance:
                # negative_image_latent = torch.zeros_like(image_latents)
                # image_latents = torch.cat([negative_image_latent, image_latents])
                
                # TODO: not using zero-valued embeddings for negatives
                image_latents = torch.cat([image_latents]*2)
            image_latents = image_latents.to(image_embeddings.dtype)

        # Repeat the image latents for each frame so we can concatenate them with the noise
        # image_latents [batch, channels, height, width] ->[batch, num_frames, channels, height, width]
        if history is not None:
            # get the updated shape
            B, num_his, C, H, W = history.shape
            num_frames_all = num_frames + num_his
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames_all, 1, 1, 1)
            if his_cond_zero:
                image_latents[:, :num_his] = 0.0 # set history to 0
        else:
            image_latents = image_latents.unsqueeze(1).repeat(1, num_frames, 1, 1, 1)
        # mask = repeat(mask, '1 h w -> 2 f 1 h w', f=num_frames)
        
        # 5. Get Added Time IDs
        added_time_ids = self._get_add_time_ids(
            fps,
            motion_bucket_id,
            noise_aug_strength,
            image_embeddings.dtype,
            batch_size,
            num_videos_per_prompt,
            do_classifier_free_guidance, 
        )
        added_time_ids = added_time_ids.to(device)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_frames,
            num_channels_latents,
            height,
            width * inp_repeat_factor,
            image_embeddings.dtype,
            device,
            generator,
            latents,
        )
        
        # NOTE: this make sure latents are from N(0,1), as this is what we assumed in predict_v
        latents = torch.randn_like(latents)
        
        # 7. Prepare guidance scale
        guidance_scale = torch.linspace(min_guidance_scale, max_guidance_scale, num_frames).unsqueeze(0)
        guidance_scale = guidance_scale.to(device, latents.dtype)
        # repeat batchwise
        guidance_scale = guidance_scale.repeat(batch_size * num_videos_per_prompt, 1)
        guidance_scale = _append_dims(guidance_scale, latents.ndim)

        self._guidance_scale = guidance_scale

        # 8. Denoising loop
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)
        
        if cond_wrist is not None:
            # if enable_uq:
            #     raise NotImplementedError('UQ is not implemented!')
            
            B, F, C, H, W = latents.shape
            cond_wrist = einops.repeat(cond_wrist, 'b l c h w -> b (f l) (n c) h w', n=3, f=num_frames) # (B, f x l, n x c , h, w)
            cond_wrist = torch.cat([cond_wrist] * 2) if do_classifier_free_guidance else cond_wrist
        
        if history is not None:
            history = torch.cat([history] * 2) if do_classifier_free_guidance else history
        
        
        if output_type == "latent":
            return_dict_solver = True
        else:
            return_dict_solver = return_dict
        
        if flow_map_type == 'flow_map':
            latents = self.flow_map_solver(
                num_inference_steps, latents,
                image_latents, image_embeddings,
                added_time_ids,
                num_his, history, cond_wrist,
                frame_level_cond, do_classifier_free_guidance, flow_map_loss_type=flow_map_loss_type,
                return_dict_solver=return_dict_solver,
            )
        elif flow_map_type == 'shortcut':
            latents = self.short_cut_solver(
                num_inference_steps, latents,
                image_latents, image_embeddings,
                added_time_ids,
                num_his, history, cond_wrist,
                frame_level_cond, do_classifier_free_guidance,
            )
        elif flow_map_type == 'flow_matching':
            latents = self.euler_solver(
                num_inference_steps, latents,
                image_latents, image_embeddings,
                added_time_ids,
                num_his, history, cond_wrist,
                frame_level_cond, do_classifier_free_guidance,
                return_dict_solver=return_dict_solver,
            )
        else:
            raise NotImplementedError(f'flow_map_type {flow_map_type} not implemented!')
            
        if not output_type == "latent":
            # cast back to fp16 if needed
            if needs_upcasting:
                self.vae.to(dtype=torch.float16)
            # latents = latents/self.vae.config.scaling_factor
            latents = latents.to(self.vae.dtype)
            frames = self.decode_latents(latents, num_frames, decode_chunk_size)
            frames = svd_tensor2vid(frames, self.video_processor, output_type=output_type)
        else:
            # it is a dict actually as return_dict_solver is true. see flow_map_solver
            frames = latents 
            latents = latents["frames"]

        self.maybe_free_model_hooks()

        if not return_dict:
            return frames, latents

        return StableVideoDiffusionPipelineOutput(frames=frames)


    @torch.no_grad()
    def predict_v(self, t, x_t, 
                  image_latents, image_embeddings, added_time_ids, num_his,
                  history=None, cond_wrist=None, distance=None, 
                  frame_level_cond = False, 
                  do_classifier_free_guidance=False):
        B, F, C, H, W =  x_t.shape

        sigma = ((1.0 - t) / t).clamp(min=0.02, max=700)
        c_in   = 1.0 / torch.sqrt(sigma**2 + 1.0)       # (B,1,1,1,1)
        c_skip = 1.0 / (sigma**2 + 1.0)
        c_out  = -sigma / torch.sqrt(sigma**2 + 1.0)
        c_noise = (sigma.log() / 4.0).view(B)           # (B,)

        # map flow point to EDM point: x_edm = x_t / t = x1 + sigma*x0
        x_t_edm = x_t / t
        
        # expand the latents if we are doing classifier free guidance
        if do_classifier_free_guidance:
            c_in = torch.cat([c_in] * 2, dim=0)
            c_noise = torch.cat([c_noise] * 2, dim=0)
        
        latent_model_input = torch.cat([x_t_edm] * 2) if do_classifier_free_guidance else x_t_edm
        latent_model_input = c_in * latent_model_input  # scale input
        
        if history is not None:
            latent_model_input = torch.cat([history, latent_model_input], dim=1) # (bsz*2,frame+F,4,32,32)

        # Concatenate image_latents over channels dimention
        latent_model_input = torch.cat([latent_model_input, image_latents], dim=2)
        
        if cond_wrist is not None:
            # print('cond_wrist_shape:',cond_wrist.shape, 'latent_model_input_shape:',latent_model_input.shape)
            latent_model_input = torch.cat([latent_model_input, cond_wrist], dim=3) # (B, 8, 12, 96, 40)
        
        latent_model_input = latent_model_input.to(self.unet.dtype)
        image_embeddings = image_embeddings.to(self.unet.dtype)
        
        if distance is not None:
            distance = torch.cat([distance] * 2) if do_classifier_free_guidance else distance
        
        noise_pred = self.unet(
            latent_model_input,
            c_noise,
            distance=distance,
            encoder_hidden_states=image_embeddings,
            added_time_ids=added_time_ids,
            return_dict=False,
            frame_level_cond=frame_level_cond,
        )[0] # torch.Size([2, 11, 4, 72, 40])

        if cond_wrist is not None:
            noise_pred = noise_pred[:, :,:,:H, :W] # remove cond_wrist
        if history is not None:
            # print('history_shape:',history.shape)
            # print('noise_pred_shape:',noise_pred.shape)
            noise_pred = noise_pred[:, num_his:, :, :, :] # remove history

        # perform guidance
        if do_classifier_free_guidance:
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + self.guidance_scale * (noise_pred_cond - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        # latents = self.scheduler.step(noise_pred, t, latents).prev_sample
        # now convert this to v-prediction in flow-map space
        x1_hat = c_out * noise_pred + c_skip * x_t_edm 
        predicted_noise = (x_t_edm - x1_hat) / sigma
        v_pred = x1_hat - predicted_noise
        
        # output
        output = {
            "pred_vel": v_pred,
            "latent_input": latent_model_input,
            "c_noise": c_noise,
            "distance": distance,
        }
        
        return output
    
    
    @torch.no_grad()
    def flow_map_solver(self, num_inference_steps, latents
                            , image_latents, image_embeddings, 
                            added_time_ids, 
                            num_his=0, history=None, cond_wrist=None, 
                            frame_level_cond = False,
                            do_classifier_free_guidance=False,
                            flow_map_loss_type='psd',
                            return_dict_solver=False):
        
        device = latents.device
        
        B = latents.shape[0]
    
        # choose shortcut scale (t close to 0 => dt close to 1 => dt_base=0)
        
        t_grid = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        t_grid = t_grid.clamp(1/(1+700), 1/(1+0.02))
        # t_grid = t_grid.clamp(1/(1+self.init_noise_sigma), 1/(1+0.02)) 
        dt_grid = t_grid[1:] - t_grid[:-1]   # (N,)
        
        if num_inference_steps == 1:
            dt_grid = torch.tensor([1.0 - t_grid[0]], device=device, dtype=torch.float32)
            print("Warning: num_inference_steps=1 for flow_map_solver, set dt to 1-t0")
        
        # t = torch.full((B,), 0, device=device, dtype=torch.float32)
        # t_ = t.view(B, 1, 1, 1, 1).clamp(1/(1+700), 1/(1+0.02))  # (B,1,1,1,1)
        # dt = 1-t_
        latent_prev = None
        t_prev = None
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                t_i = t_grid[i]  
                dt = dt_grid[i] 
                 
                t_ = t_i.expand(B).view(B, 1, 1, 1, 1)
                dt = dt.expand(B).view(B, 1, 1, 1, 1)
                if flow_map_loss_type == 'flow_matching':
                    dt = torch.zeros_like(dt)
                
                v_pred_output = self.predict_v(t_, latents, 
                                        image_latents, image_embeddings, 
                                        added_time_ids, num_his,
                                        history=history, cond_wrist=cond_wrist, distance=dt, 
                                        frame_level_cond=frame_level_cond, 
                                        do_classifier_free_guidance=do_classifier_free_guidance)
                
                # get pred velocity
                v_pred = v_pred_output["pred_vel"]
                
                latents_prev = latents
                latents = latents + v_pred * dt
            
                progress_bar.update()
        
        if not return_dict_solver:
            return latents
        else:
            if do_classifier_free_guidance:
                latent_model_input_wo_cond = v_pred_output["latent_input"].chunk(2)[0]
                v_pred_output["c_noise"] = v_pred_output["c_noise"].chunk(2)[0]
                v_pred_output["distance"] = v_pred_output["distance"].chunk(2)[0]
            else:
                latent_model_input_wo_cond = v_pred_output["latent_input"]

            # output
            output = {
                "frames": latents,
                "unet_input": latent_model_input_wo_cond,
                "timestep": v_pred_output["c_noise"].view(B),
                "distance": v_pred_output["distance"].view(B),
            }

            return output

    @torch.no_grad()
    def euler_solver(self, num_inference_steps, latents,
                     image_latents, image_embeddings,
                     added_time_ids,
                     num_his=0, history=None, cond_wrist=None,
                     frame_level_cond=False,
                     do_classifier_free_guidance=False,
                     return_dict_solver=False):
        B = latents.shape[0]
        device = latents.device

        t_grid = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)
        dt_grid = t_grid[1:] - t_grid[:-1]

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for i in range(num_inference_steps):
                t_i = t_grid[i]
                dt = dt_grid[i]

                t_ = t_i.expand(B).view(B, 1, 1, 1, 1).clamp(1/(1+700), 1/(1+0.02))

                # Pure-Euler flow matching: pass zero "distance" so checkpoints
                # trained with distance-conditioning still get a valid tensor
                # (Timesteps proj at 0 → no shortcut bias).
                distance = torch.zeros(B, device=device, dtype=torch.float32)

                v_pred_output = self.predict_v(t_, latents,
                                               image_latents, image_embeddings,
                                               added_time_ids, num_his,
                                               history=history, cond_wrist=cond_wrist, distance=distance,
                                               frame_level_cond=frame_level_cond,
                                               do_classifier_free_guidance=do_classifier_free_guidance)
                v_pred = v_pred_output["pred_vel"]

                latents = latents + v_pred * dt

                progress_bar.update()

        if not return_dict_solver:
            return latents

        if do_classifier_free_guidance:
            latent_model_input_wo_cond = v_pred_output["latent_input"].chunk(2)[0]
            v_pred_output["c_noise"] = v_pred_output["c_noise"].chunk(2)[0]
            v_pred_output["distance"] = v_pred_output["distance"].chunk(2)[0]
        else:
            latent_model_input_wo_cond = v_pred_output["latent_input"]

        return {
            "frames": latents,
            "unet_input": latent_model_input_wo_cond,
            "timestep": v_pred_output["c_noise"].view(B),
            "distance": v_pred_output["distance"].view(B),
        }

    # from Fast-Ctrl-World
    @torch.no_grad()
    def short_cut_solver(self, num_inference_steps, latents,
                         image_latents, image_embeddings,
                         added_time_ids,
                         num_his=0, history=None, cond_wrist=None,
                         frame_level_cond=False,
                         do_classifier_free_guidance=False):
        device = latents.device
        B = latents.shape[0]

        if num_inference_steps <= 1:
            # Single-step shortcut
            dt_base_val = 0
            dt = 2.0 ** (-dt_base_val)   # = 1
            t_val = 1.0 - dt             # = 0

            t = torch.full((B,), t_val, device=device, dtype=torch.float32)
            t_ = t.view(B, 1, 1, 1, 1).clamp(1/(1+700), 1/(1+0.02))
            distance = torch.full((B,), dt_base_val, device=device, dtype=torch.int64)

            with self.progress_bar(total=1) as progress_bar:
                v_pred_output = self.predict_v(t_, latents,
                                               image_latents, image_embeddings,
                                               added_time_ids, num_his,
                                               history=history, cond_wrist=cond_wrist, distance=distance,
                                               frame_level_cond=frame_level_cond,
                                               do_classifier_free_guidance=do_classifier_free_guidance)
                v_pred = v_pred_output["pred_vel"]
                latents = latents + (1 - t_) * v_pred
                progress_bar.update()
        else:
            # Multi-step shortcut with distance conditioning
            t_grid = torch.linspace(0.0, 1.0, num_inference_steps + 1, device=device, dtype=torch.float32)
            dt_step = t_grid[1] - t_grid[0]  # = 1/num_inference_steps

            dt_base_val = int(math.log2(num_inference_steps)) + 1
            distance = torch.full((B,), dt_base_val, device=device, dtype=torch.int64)

            with self.progress_bar(total=num_inference_steps) as progress_bar:
                for i in range(num_inference_steps):
                    t_i = t_grid[i]
                    t_ = t_i.expand(B).view(B, 1, 1, 1, 1).clamp(1/(1+700), 1/(1+0.02))

                    v_pred_output = self.predict_v(t_, latents,
                                                   image_latents, image_embeddings,
                                                   added_time_ids, num_his,
                                                   history=history, cond_wrist=cond_wrist, distance=distance,
                                                   frame_level_cond=frame_level_cond,
                                                   do_classifier_free_guidance=do_classifier_free_guidance)
                    v_pred = v_pred_output["pred_vel"]
                    latents = latents + dt_step * v_pred
                    progress_bar.update()

        return {"frames": latents}