import os
import torch
import argparse
from PIL import Image

from diffusers.models.autoencoders.autoencoder_kl_hunyuan_video import AutoencoderKLHunyuanVideo
from diffusers.pipelines.hunyuan_video.pipeline_hunyuan_video import HunyuanVideoPipeline
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler

from transformer_hunyuan_video_mocha import HunyuanVideoTransformer3DModelMoCha
from pipeline_hunyuan_video_mocha import HunyuanVideoMoChaPipeline
from embed_audio import Wav2VecEmbedder
from utils import export_video_with_optional_audio, load_model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--num_frames", type=int, default=129)
    p.add_argument("--fps", type=int, default=24)
    p.add_argument("--audio_path", type=str, required=True)
    p.add_argument(
        "--task",
        type=str,
        choices=["st2v", "sti2v"],
        required=True,
        help="Generation task: st2v (speech+text→video) or sti2v (speech+image+text→video)",
    )
    p.add_argument("--i2v_img_path", type=str, default=None,
                   help="The first frame. Required only for sti2v")
    p.add_argument("--output_path", type=str, required=True)
    p.add_argument("--transformer_ckpt_path", type=str, required=True, help="MoCha model ckpt local path")
    return p.parse_args()


def main():
    args = parse_args()
    if args.task == "sti2v" and args.i2v_img_path is None:
        raise ValueError("sti2v requires --i2v_img_path")

    hunyuan_model_id = "hunyuanvideo-community/HunyuanVideo"

    # Create a dummy pipeline as the txt_encoder, this txt_encoder will run on cpu only.
    txt_encoder = HunyuanVideoPipeline.from_pretrained(
        hunyuan_model_id,
        device_map=None, 
    )
    for name, module in txt_encoder.components.items():
        if hasattr(module, "eval"):
            module.eval()
        
    # Load HunyuanVideo VAE
    vae = AutoencoderKLHunyuanVideo.from_pretrained(
        hunyuan_model_id,
        subfolder="vae", 
        low_cpu_mem_usage=False,  
        device_map=None 
    )
    vae.eval()
    
    # Load scheduler
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        hunyuan_model_id,
        subfolder="scheduler"
    )

     # Load HunyuanVideo transformer
    transformer = HunyuanVideoTransformer3DModelMoCha.from_pretrained(
        hunyuan_model_id,
        subfolder="transformer",
        low_cpu_mem_usage=False,  # Avoid meta tensors
        device_map=None,  # Let us handle device placement manually
    )

    # Re-initialize S2V-specific layers after loading pretrained HunyuanVideo weights
    transformer._init_s2v_layers()
    print("[INIT] Re-initialized S2V-specific layers after loading HunyuanVideo pretrained weights")

    # Load Hunyuan MoCha ckpt from local
    def rename_func(state_dict):
            new_state_dict = {}
            for k, v in state_dict.items():
                # remove leading "transformer." if present
                new_k = k.replace("transformer.", "", 1) if k.startswith("transformer.") else k
                new_state_dict[new_k] = v
            return new_state_dict
    if isinstance(args.transformer_ckpt_path, str):
        print(f"[INIT] loading ckpt from {args.transformer_ckpt_path}")
        transformer = load_model(transformer, args.transformer_ckpt_path, rename_func=rename_func)

    # Build Hunyuan MoCha pipeline
    pipeline = HunyuanVideoMoChaPipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        txt_encoder=txt_encoder,
    ).to(device="cuda", dtype=torch.bfloat16)

    # Load wav2vec2 embedder
    embedder = Wav2VecEmbedder(device="cuda")
    audio_embed = embedder.embed(
        audio_path=args.audio_path,
    )
    print(f"audio_embed.shape:{audio_embed.shape}")  # [129, 12, 768]

    # Sample audio embedding
    t_audio = audio_embed.shape[0]
    t_available = min(t_audio, args.num_frames)
    # Temporal chunking: (num_frames-1)//temporal_unit_size*temporal_unit_size+1
    # hunyuanvideo must be N * 4 + 1
    temporal_unit_size = 4
    t_chunked = (t_available - 1) // temporal_unit_size * temporal_unit_size + 1
    start_idx = 0
    end_idx = start_idx + t_chunked
    temporal_indices = list(range(start_idx, end_idx))
    audio_embed_sampled = audio_embed[temporal_indices]  # [t_chunked, 12, 768]

    # Generation
    prompt = "The video features a medium close-up shot of a young man with a fair complexion, standing on a softly lit stage. He is dressed in a crisp white shirt and a loosely knotted black tie, holding a microphone in his right hand as he sings. His expression is calm and focused, with slightly parted lips and attentive eyes that convey a sense of emotional engagement with the performance. Subtle studio lights behind him create a gentle rim glow around his silhouette, while the blurred shape of a white grand piano in the background adds depth and atmosphere. His posture is upright and composed, suggesting confidence and sincerity as he delivers the song, with the lighting enhancing the warm and intimate tone of the scene."
    negative_prompt="Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, weak dynamics, erratic motions, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards." 
    pipeline_kwargs = dict(
        prompts=[prompt],
        negative_prompt=negative_prompt,
        audio_embeds=[audio_embed_sampled],
        height=480,
        width=832,
        num_frames=t_chunked,
        num_inference_steps=50,
        seed=42,
        guidance_scale=7.0,
        audio_guidance_scale=2.0,
        timestep_shift=7.0,
        task=args.task,
    )
    if args.task == "sti2v":
        i2v_image = Image.open(args.i2v_img_path).convert("RGB")
        pipeline_kwargs["i2v_images"] = [i2v_image]
    output = pipeline(**pipeline_kwargs).frames[0]

    # Save
    export_video_with_optional_audio(
        frames=output,
        output_path=args.output_path,
        fps=args.fps,
        audio_path=args.audio_path,
    )

if __name__ == "__main__":
    main()