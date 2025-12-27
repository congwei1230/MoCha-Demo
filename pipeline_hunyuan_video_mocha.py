"""
HunyuanVideo MoCha Pipeline with Flow Matching.
"""

from ast import Raise
import inspect
import math
import time
import numpy as np
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, Union, Literal

import torch
from torchvision import transforms
import PIL.Image
from PIL import Image
from diffusers import DiffusionPipeline
from diffusers.pipelines.pipeline_utils import ImagePipelineOutput
from diffusers.utils import deprecate
from diffusers.video_processor import VideoProcessor
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from einops import rearrange

import os
import torch

from diffusers.pipelines.hunyuan_video.pipeline_output import HunyuanVideoPipelineOutput
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipeline
from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo

from transformer_hunyuan_video_mocha import HunyuanVideoTransformer3DModelMoCha


def pad_to_target_shape(tensor, target_shape):
    padding = []  # [w1, w2, h1, h2, f1, f2, c1, c2, b1, b2]
    for current, target in zip(tensor.shape, target_shape):
        padding = [0, target - current] + padding
    padded_tensor = torch.nn.functional.pad(tensor, padding)
    mask = torch.ones_like(tensor[:, :1], dtype=tensor.dtype)
    padded_mask = torch.nn.functional.pad(mask, padding, value=0)
    return padded_tensor, padded_mask

def pack_data(data):
    sizes = [t.size() for t in data]
    _, c, max_f, max_h, max_w = [max(sizes_dim) for sizes_dim in zip(*sizes)]
    res, mask = [], []
    for ten in data:
        ten, m = pad_to_target_shape(ten, [1, c, max_f, max_h, max_w])
        res.append(ten)
        mask.append(m)
    return torch.cat(res, 0), torch.cat(mask, 0)


class HunyuanVideoMoChaPipeline(DiffusionPipeline):
    """
    HunyuanVideo MoCha Pipeline
    """

    def __init__(
        self,
        # HunyuanVideo components
        transformer: HunyuanVideoTransformer3DModelMoCha,
        vae: AutoencoderKLHunyuanVideo,
        scheduler: FlowMatchEulerDiscreteScheduler,
        # llm text encoder
        txt_encoder: HunyuanVideoPipeline,
    ):
        super().__init__()
        
        # Register all pipeline components
        self.register_modules(
            transformer=transformer,
            vae=vae,
            scheduler=scheduler,
            txt_encoder=txt_encoder,
        )
        
        # Set up VAE scale factors (from HunyuanVideo)
        self.vae_scale_factor_temporal = self.vae.temporal_compression_ratio if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.spatial_compression_ratio if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    @torch.no_grad()
    def online_i2v_img_VAE_encode(self, i2v_img_pixel_values): 
        assert isinstance(i2v_img_pixel_values, list) and i2v_img_pixel_values[0].dim() == 5  # i2v_img_pixel_values: [(1 f c h w)]
        latents = [self.video2latents(
            i2v_img_pixel_value, 
            in_pattern="b f c h w", 
            out_pattern="b c f h w") for i2v_img_pixel_value in i2v_img_pixel_values]
        latents, masks = pack_data(latents)
        return latents, masks

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 32,
        height: int = 720,
        width: int = 1280,
        num_frames: int = 129,
        dtype: Optional[torch.dtype] = None,
        device: Optional[torch.device] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        print(f"num_frames: {num_frames}")
        print(f"height: {height}")
        print(f"width: {width}")
        shape = (
            batch_size,
            num_channels_latents,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents


    @torch.no_grad()
    def video2latents(self, video, in_pattern="b f c h w", vae_pattern="b c f h w", out_pattern="b c f h w"):
        assert video.ndim == 5, f"Expected 5D video, got {video.shape}"
        batch_size = video.shape[0]
        video = video.to(self.vae.device, self.vae.dtype)

        # Sanity checks so einops won't scramble C/F
        if in_pattern == "b f c h w":
            # interpret dim2 as channels
            assert video.shape[2] in (1, 3), f"Expected channels in dim=2 for '{in_pattern}', got shape {video.shape}"
        elif in_pattern == "b c f h w":
            assert video.shape[1] in (1, 3), f"Expected channels in dim=1 for '{in_pattern}', got shape {video.shape}"
        else:
            raise ValueError(f"Unsupported in_pattern: {in_pattern}")
        
        video = rearrange(video, f"{in_pattern} -> {vae_pattern}", b=batch_size)
        latents = self.vae.encode(video).latent_dist.sample()
        latents = latents * self.vae.config.scaling_factor
        latents = rearrange(latents, f"{vae_pattern} -> {out_pattern}", b=batch_size)
        return latents
    
    @torch.no_grad() # TODO: fix this
    def get_prompt_embeddings(self, prompts, device=None, dtype=None):
        """
        llm tokenizing + llm encoding
        
        Args:
            prompts: List of text prompts
            images: [[PIL.Image.Image,...] x b]
            videos: [[torch.tensor (f h w c) 0-255] x b]
            device: Target device
            dtype: Target dtype
        """
        if prompts is None:
            raise ValueError("prompts must be provided")
        prompt_embeds, pooled_prompt_embeds, prompt_attention_mask = self.txt_encoder.encode_prompt(
            prompt=prompts,
        )
        return prompt_embeds.to(dtype=dtype, device=device), pooled_prompt_embeds.to(dtype=dtype, device=device), prompt_attention_mask.to(device)

    def pil_to_i2v_tensor(self, pil: Image.Image, device=None, dtype=None):
        if pil.mode != "RGB":
            pil = pil.convert("RGB")

        arr = np.asarray(pil)  # [H, W, 3], uint8
        x = torch.from_numpy(np.ascontiguousarray(arr)).float()  # [H, W, 3]
        x = x.permute(2, 0, 1).unsqueeze(0).unsqueeze(0)  # [1, 1, 3, H, W]

        # Normalize
        x = x / 127.5 - 1.0

        if device is not None or dtype is not None:
            x = x.to(device=device or x.device, dtype=dtype or x.dtype)
        return x
    
    @torch.no_grad()
    def __call__(
        self,
        prompts: Union[str, List[str]] = None,
        audio_embeds: Union[None, List] = None, # [(T, 12, 768),...]
        i2v_images: Union[None, List] = None,  #  [PIL.Image.Image, ...]
        task: str = "", # st2v or sti2v
        negative_prompt: str = "",
        num_inference_steps: int = 30,  # HunyuanVideo default
        timesteps: List[int] = None,
        guidance_scale: float = 6.0,  # HunyuanVideo default
        audio_guidance_scale: float = 1.5,
        num_images_per_prompt: Optional[int] = 1,
        num_frames: int = 129,  # HunyuanVideo default
        fps: float = 15.0,  # HunyuanVideo default
        cond_fps: Optional[float] = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        cond_height: Optional[int] = None,
        cond_width: Optional[int] = None,
        generator: Optional[Union[torch.Generator, List[torch.Generator]]] = None,
        latents: Optional[torch.FloatTensor] = None,
        prompt_embeds: Optional[torch.FloatTensor] = None,
        prompt_attention_mask: Optional[torch.FloatTensor] = None,
        negative_prompt_embeds: Optional[torch.FloatTensor] = None,
        negative_prompt_attention_mask: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pt",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: int = 1,
        clean_caption: bool = True,
        timestep_shift: Optional[float] = None,
        **kwargs,
    ) -> Union[ImagePipelineOutput, Tuple]:
        
        SUPPORTED_TASKS = {"st2v", "sti2v"}

        assert task in SUPPORTED_TASKS, \
            f"Unsupported task {task}, must be one of {SUPPORTED_TASKS}"

        if task == "st2v":
            assert not i2v_images, "st2v should not receive i2v_images"

        if task == "sti2v":
            assert i2v_images, "sti2v requires i2v_images"

        if "process_call_back" in kwargs:
            process_call_back = kwargs["process_call_back"]
        else:
            process_call_back = None

        # 1. Check inputs and set defaults
        height = height or self.transformer.config.sample_size * self.vae_scale_factor
        width = width or self.transformer.config.sample_size * self.vae_scale_factor
        cond_height = cond_height or height
        cond_width = cond_width or width
        # cond_num_frames = cond_num_frames or num_frames
        cond_fps = cond_fps or fps
        timestep_shift = timestep_shift

        # TODO: check check_inputs
        # 2. Batch size
        if prompts is not None and isinstance(prompts, str):
            batch_size = 1
        elif prompts is not None and isinstance(prompts, list):
            batch_size = len(prompts)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device

        # 3. Classifier free guidance
        do_text_cfg = guidance_scale > 1.0
        do_img_cfg = audio_guidance_scale > 1.0
        print(f"do_text_cfg:{do_text_cfg}")
        print(f"do_img_cfg:{do_img_cfg}")
        print(f"negative_prompt:{negative_prompt} ")

        # 5. Prepare latents
        latent_channels = self.transformer.config.in_channels

        # handle audio
        audio_embeds = torch.stack(audio_embeds, dim=0)   # (B, T, 12, 768)
        audio_embeds = audio_embeds.to(dtype=self.dtype, device=device)
        assert audio_embeds.shape[0] == 1, f"Does not support bs > 1 for now, the following code only work for BS=1"
        uncond_audio_embeds = torch.zeros_like(audio_embeds, dtype=self.dtype, device=device)
        print(f"__call__ audio_embeds.shape: {audio_embeds.shape}")
        num_frames = audio_embeds.shape[1]
        print(f"__call__num_frames: {num_frames}")

        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            latent_channels,
            height,
            width,
            num_frames,
            self.dtype,
            device,
            generator,
            latents,
        )
        batch_size,  _, latent_t, latent_h, latent_w = latents.shape        
        print(f"building latents.shape: {latents.shape}")


        # Add condition
        attention_mask = torch.ones_like(latents[:, :1], dtype=latents.dtype) # (b, 1, f, h, w)
        
        assert batch_size == 1, f"Does not support bs > 1 for now"  # TODO: fix this
        is_cond = torch.zeros(latent_t, dtype=torch.bool, device=latents.device)
        
        # STI2V task
        if  task == "sti2v" and i2v_images is not None:
            if isinstance(i2v_images, Image.Image):
                i2v_images = [i2v_images]

            # normalized [(1 f c h w), (1, 1, 3, 352, 704), ...,]
            i2v_img_pixel_values = [
                self.pil_to_i2v_tensor(pil, device=device, dtype=self.dtype)  # or self.transformer.dtype
                for pil in i2v_images
            ]
            image_h, image_w = latent_h * self.vae_scale_factor_spatial, latent_w * self.vae_scale_factor_spatial
            i2v_img_pixel_values_resized = []
            for i2v_img_pixel_value in i2v_img_pixel_values:  # shape (1, f, c, h, w)
                _, f, c, h, w = i2v_img_pixel_value.shape
                i2v_img_pixel_value = i2v_img_pixel_value.view(1 * f, c, h, w)  # (1*f, c, h, w)
                import torch.nn.functional as F
                resized = F.interpolate(
                    i2v_img_pixel_value,
                    size=(image_h, image_w),
                    mode="bicubic",
                    align_corners=False
                )
                resized = resized.view(1, f, c, image_h, image_w)  # back to (1, f, c, H, W)
                i2v_img_pixel_values_resized.append(resized)
            i2v_img_latents, i2v_img_attn_mask = self.online_i2v_img_VAE_encode(i2v_img_pixel_values_resized)  # b c f h w

            # replace the first latent with starting image
            latents[:, :, 0:1] = i2v_img_latents
            attention_mask[:, :, 0:1] = i2v_img_attn_mask
            is_cond[0] = True
            print(f"[DEBUG] STI2V task, latents.shape: {latents.shape}")

        assert is_cond.shape[0] == latents.shape[2], "full latents should match with is_cond over f dimension"

        # 4. Encode input prompt with LLM and CLIP
        prompt_embeds_uncond, pooled_prompt_embeds_uncond,prompt_attention_mask_uncond = self.get_prompt_embeddings(
            prompts=negative_prompt,
            device=device,
            dtype=self.transformer.dtype
        )
        prompt_embeds_txt, pooled_prompt_embeds_txt, prompt_attention_mask_txt = self.get_prompt_embeddings(
            prompts=prompts,
            device=device,
            dtype=self.transformer.dtype
        )

        idx_no_cond = (~is_cond).nonzero(as_tuple=False).squeeze(-1)      # [T_keep]
        assert idx_no_cond.numel() > 0, "All f dimension are conditioned; nothing to train."

        # 7. Denoising loop
        timesteps_all = torch.linspace(1.0, 0, num_inference_steps + 1, device=latents.device)
        timesteps_all = timestep_shift * timesteps_all / (1 - timesteps_all + timestep_shift * timesteps_all)
        dts = timesteps_all[:-1] - timesteps_all[1:]
        timesteps = timesteps_all[:-1]

        # TODO: right now can't handle batch size > 1
        assert batch_size == 1
        latents_full_origin = latents.clone() 
        latents_full = latents

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            t0 = time.time()
            for i, t in enumerate(timesteps):
                guidance_tensor = torch.tensor([6.0], device=device) * 1000.0 # Guidance tensor (HunyuanVideo scales by 1000)
                
                current_timestep = t

                if not torch.is_tensor(current_timestep):
                    is_mps = latents_full.device.type == "mps"
                    if isinstance(current_timestep, float):
                        dtype = torch.float32 if is_mps else torch.float64
                    else:
                        dtype = torch.int32 if is_mps else torch.int64
                    current_timestep = torch.tensor([current_timestep], dtype=dtype, device=latents_full.device)
                elif len(current_timestep.shape) == 0:
                    current_timestep = current_timestep[None].to(latents_full.device)
                
                current_timestep = current_timestep.expand(latents_full.shape[0])

                # 3 pass
                if guidance_scale > 1.0 and audio_guidance_scale > 1.0:
                    print(f"[DEBUG 3 pass")
                    v_pred_uncond= self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_uncond,
                        encoder_attention_mask=prompt_attention_mask_uncond,
                        pooled_projections=pooled_prompt_embeds_uncond,
                        audio_embeds=uncond_audio_embeds,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_uncond = v_pred_uncond.index_select(2, idx_no_cond)
                    v_pred_negtxt_audio = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_uncond,
                        encoder_attention_mask=prompt_attention_mask_uncond,
                        pooled_projections=pooled_prompt_embeds_uncond,
                        audio_embeds=audio_embeds,                        
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_negtxt_audio = v_pred_negtxt_audio.index_select(2, idx_no_cond)
                    v_pred_txt_audio = self.transformer(
                        hidden_states=latents_full,                   # [1,C,T,H,W]
                        timestep=current_timestep * 999,             # match original scaling
                        encoder_hidden_states=prompt_embeds_txt,
                        encoder_attention_mask=prompt_attention_mask_txt,
                        pooled_projections=pooled_prompt_embeds_txt,
                        audio_embeds=audio_embeds,
                        guidance=guidance_tensor,
                        return_dict=False,
                    )[0]
                    v_pred_txt_audio = v_pred_txt_audio.index_select(2, idx_no_cond)
                    v_pred = 7 * v_pred_txt_audio - 3 * v_pred_negtxt_audio - 3 * v_pred_uncond
                else:
                    raise ValueError(f"guidance_scale: {guidance_scale} and audio_guidance_scale:{audio_guidance_scale} is not support") 

                # compute previous image: x_t -> x_t-1
                latents_no_cond = latents_full.index_select(2, idx_no_cond)            # [B,C,T_keep,H,W]
                print(f"latents_full.shape:{latents_full.shape}")
                print(f"v_pred.shape:{v_pred.shape}")
                print(f"dts[i]:{dts[i]}")
                latents_no_cond = latents_no_cond - dts[i] * v_pred

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        step_idx = i // getattr(self.scheduler, "order", 1)
                        callback(step_idx, t, latents_no_cond)

                if process_call_back:
                    process_call_back((i + 1) / len(timesteps), (time.time() - t0) / (i + 1) * (len(timesteps) - i - 1))

                # Reset the latents full from origin
                latents_full = latents_full_origin.clone()
                latents_full.index_copy_(2, idx_no_cond, latents_no_cond)

        # 8. Decode latents
        if task == "sti2v":
            # I2V task we keep the ref frame during decoding
            is_cond[0] = False
            idx_no_cond = (~is_cond).nonzero(as_tuple=False).squeeze(-1)

        latents_no_cond = latents_full.index_select(2, idx_no_cond)
        latents = latents_no_cond
        if not output_type == "latent":
            latents = latents.to(self.vae.dtype) / self.vae.config.scaling_factor
            video = self.vae.decode(latents, return_dict=False)[0]
            print(f"video.shape: {video.shape}, type: {type(video)}")
            print(f"min: {video.min()}, max: {video.max()}, dtype: {video.dtype}")

            # video = self.video_processor.postprocess_video(video, output_type=output_type)
            video = self.video_processor.postprocess_video(video, output_type="np")

            # video.shape: (1, 77, 256, 256, 3), type: <class 'numpy.ndarray'>
            # [b, t, h, w, c]
            # min: 0.001953125, max: 0.984375, dtype: float32
            print(f"video.shape: {video.shape}, type: {type(video)}")
            print(f"min: {video.min()}, max: {video.max()}, dtype: {video.dtype}")
        else:
            video = latents

        # Offload all models
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return HunyuanVideoPipelineOutput(frames=video)