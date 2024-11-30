"""
This script demonstrates how to generate a video from a text prompt using CogVideoX with modern quantization
and memory optimizations for consumer GPUs.
"""

import argparse
import os
import torch
import torch._dynamo
from diffusers import AutoencoderKLCogVideoX, CogVideoXTransformer3DModel, CogVideoXPipeline, CogVideoXDPMScheduler
from diffusers.utils import export_to_video
from transformers import T5EncoderModel
from torchao.quantization import autoquant, DEFAULT_INT4_AUTOQUANT_CLASS_LIST

# Configure PyTorch settings
os.environ["TORCH_LOGS"] = "+dynamo,output_code,graph_breaks,recompiles"
torch._dynamo.config.suppress_errors = True
torch.set_float32_matmul_precision("high")
torch._inductor.config.conv_1x1_as_mm = True
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.epilogue_fusion = False
torch._inductor.config.coordinate_descent_check_all_directions = True

def setup_model(model_path, dtype=torch.float16, device="cuda"):
    """
    Load and set up the CogVideoX pipeline with appropriate optimizations
    """
    # Load pipeline with CPU offload by default
    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    )
    
    # Set up scheduler
    pipe.scheduler = CogVideoXDPMScheduler.from_config(
        pipe.scheduler.config, 
        timestep_spacing="trailing"
    )
    
    return pipe

def apply_optimizations(pipe, memory_mode='medium'):
    """
    Apply memory optimizations based on the selected mode
    """
    if memory_mode == 'low':
        # Sequential CPU offload for maximum memory savings
        pipe.enable_sequential_cpu_offload()
    elif memory_mode == 'medium':
        # Balanced approach with model CPU offload and VAE optimizations
        pipe.enable_model_cpu_offload()
        pipe.vae.enable_tiling()
        pipe.vae.enable_slicing()
    else:  # 'high'
        # Minimum memory optimization
        pipe.enable_model_cpu_offload()

    # Apply channels last memory format where possible
    if hasattr(pipe.transformer, 'to'):
        pipe.transformer.to(memory_format=torch.channels_last)

    return pipe

def generate_video(
    prompt: str,
    model_path: str,
    output_path: str = "./output.mp4",
    num_inference_steps: int = 50,
    guidance_scale: float = 6.0,
    num_videos_per_prompt: int = 1,
    memory_mode: str = 'medium',
    dtype: torch.dtype = torch.bfloat16,
    num_frames: int = 49,
    fps: int = 8,
    seed: int = 42,
    device: str = "cuda"
):
    """
    Generates a video based on the given prompt using memory optimizations.
    """
    try:
        # Set up the pipeline with appropriate optimizations
        pipe = setup_model(model_path, dtype, device)
        pipe = apply_optimizations(pipe, memory_mode)
        
        # Generate video
        video = pipe(
            prompt=prompt,
            num_videos_per_prompt=num_videos_per_prompt,
            num_inference_steps=num_inference_steps,
            num_frames=num_frames,
            use_dynamic_cfg=True,
            guidance_scale=guidance_scale,
            generator=torch.Generator(device=device).manual_seed(seed),
        ).frames[0]
        
        # Export video
        export_to_video(video, output_path, fps=fps)
        print(f"Video successfully generated and saved to {output_path}")
        
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("\nGPU out of memory error. Try one of the following:")
            print("1. Use --memory_mode low for maximum memory savings")
            print("2. Reduce --num_frames (current: {num_frames})")
            print("3. Use a smaller model variant (e.g., CogVideoX-2b instead of CogVideoX-5b)")
        raise e
    finally:
        # Clean up
        if 'pipe' in locals():
            del pipe
            torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate a video from a text prompt using CogVideoX")
    parser.add_argument("--prompt", type=str, required=True, help="The description of the video to be generated")
    parser.add_argument("--model_path", type=str, default="THUDM/CogVideoX-5b", help="Path of the pre-trained model")
    parser.add_argument("--output_path", type=str, default="./output.mp4", help="Path to save generated video")
    parser.add_argument("--num_inference_steps", type=int, default=50, help="Inference steps")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Classifier-free guidance scale")
    parser.add_argument("--num_videos_per_prompt", type=int, default=1, help="Videos to generate per prompt")
    parser.add_argument("--dtype", type=str, default="bfloat16", help="Data type (e.g., 'float16', 'bfloat16')")
    parser.add_argument(
        "--memory_mode",
        type=str,
        choices=['low', 'medium', 'high'],
        default='medium',
        help="Memory optimization mode: low (~4GB), medium (~11GB), or high (~19GB)"
    )
    parser.add_argument("--num_frames", type=int, default=49, help="Number of frames in the video")
    parser.add_argument("--fps", type=int, default=8, help="Frames per second for output video")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cpu)")

    args = parser.parse_args()
    dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
    
    generate_video(
        prompt=args.prompt,
        model_path=args.model_path,
        output_path=args.output_path,
        num_inference_steps=args.num_inference_steps,
        guidance_scale=args.guidance_scale,
        num_videos_per_prompt=args.num_videos_per_prompt,
        memory_mode=args.memory_mode,
        dtype=dtype,
        num_frames=args.num_frames,
        fps=args.fps,
        seed=args.seed,
        device=args.device
    )
