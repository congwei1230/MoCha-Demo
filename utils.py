from diffusers.utils import export_to_video
from moviepy import VideoFileClip, AudioFileClip
import os
import torch


def load_model(model, ckpt_path, rename_func=None):
    """Load a checkpoint into a model by copying matching named_parameters.
    Prints missing/unexpected keys and any copy errors. Returns the model."""
    print(f"Loading model {type(model)} from checkpoint: " + ckpt_path)
    state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if rename_func is not None:
        state_dict = rename_func(state_dict)
    for name, param in model.named_parameters():
        if name in state_dict:
            try:
                param.data.copy_(state_dict[name])
            except RuntimeError as e:
                print(f"Error loading {name}: {e}")
            state_dict.pop(name)
        else:
            print(f"Missing in state_dict: {name}")
    if len(state_dict) > 0:
        for name in state_dict:
            print(f"Unexpected in state_dict: {name}")
    return model

def export_video_with_optional_audio(
    frames,
    output_path: str,
    fps: int = 24,
    audio_path: str | None = None,
    tmp_suffix: str = "_noaudio",
    codec: str = "libx264",
    audio_codec: str = "aac",
    verbose: bool = True,
):
    """
    Export frames to MP4, optionally mux audio, and cleanup temp files.

    Args:
        frames: np.ndarray or torch.Tensor shaped (T, H, W, 3).
               Values may be float in [0,1] or [0,255], depending on pipeline.
        output_path: final mp4 path.
        fps: frames per second.
        audio_path: optional audio file path (mp3/wav/etc). If None, exports silent video.
        tmp_suffix: suffix for intermediate silent video when muxing audio.
        codec/audio_codec: ffmpeg codec names used by moviepy for final write.
        verbose: prints basic stats about frames.
    """
    # Debug / sanity prints (optional)
    if verbose:
        try:
            vmin = frames.min()
            vmax = frames.max()
        except Exception:
            vmin = vmax = "N/A"
        try:
            dtype = frames.dtype
        except Exception:
            dtype = type(frames)
        print(f"data.shape: {getattr(frames, 'shape', None)}, type: {type(frames)}")
        print(f"min: {vmin}, max: {vmax}, dtype: {dtype}")

    # Decide where to export the raw video
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if audio_path is None:
        export_to_video(frames, output_path, fps=fps)
        return output_path

    root, ext = os.path.splitext(output_path)
    tmp_video_path = f"{root}{tmp_suffix}{ext}"

    # 1) export silent video first
    export_to_video(frames, tmp_video_path, fps=fps)

    # 2) mux audio
    video_clip = VideoFileClip(tmp_video_path)
    audio_clip = AudioFileClip(audio_path)

    # Trim audio to video duration (or leave shorter as-is)
    if audio_clip.duration > video_clip.duration:
        audio_clip = audio_clip.subclipped(0, video_clip.duration)

    video_with_audio = video_clip.with_audio(audio_clip)

    # 3) Write final video with audio
    video_with_audio.write_videofile(
        output_path,
        fps=fps,
        codec=codec,
        audio_codec=audio_codec,
    )

    # cleanup
    video_clip.close()
    audio_clip.close()
    video_with_audio.close()

    if os.path.exists(tmp_video_path):
        os.remove(tmp_video_path)

    return output_path