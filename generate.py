# Default Settings (Mostly T2I oriented, override via command line)
DEFAULT_MODEL = "SG161222/RealVisXL_V4.0" # Default to a T2I XL model
DEFAULT_VARIANT = None
DEFAULT_CUSTOM_PIPELINE = None
DEFAULT_SCHEDULER = "EulerAncestralDiscreteScheduler"
DEFAULT_LORA = None
DEFAULT_CONTROLNET = None
DEFAULT_STEPS = 30
DEFAULT_PROMPT = "best quality, realistic, unreal engine, 4K, a cat sitting on human lap"
DEFAULT_NEGATIVE_PROMPT = ""
DEFAULT_SEED = 333
DEFAULT_WARMUPS = 1
DEFAULT_BATCH = 1
DEFAULT_HEIGHT = None # Auto-detect from model if None
DEFAULT_WIDTH = None  # Auto-detect from model if None
DEFAULT_INPUT_IMAGE = None # If provided, triggers I2I mode
DEFAULT_CONTROL_IMAGE = None
DEFAULT_OUTPUT_IMAGE = "generated_image.png" # Provide a default output name
DEFAULT_EXTRA_CALL_KWARGS = None # e.g., '{"strength": 0.75, "guidance_scale": 7.5}'
DEFAULT_CACHE_INTERVAL = 3
DEFAULT_CACHE_LAYER_ID = 0
DEFAULT_CACHE_BLOCK_ID = 0
DEFAULT_COMPILER = "nexfort"
DEFAULT_COMPILER_CONFIG = None
DEFAULT_QUANTIZE_CONFIG = None
DEFAULT_TASK = "auto" # Can be 'auto', 'text2image', 'image2image', 'instructpix2pix'

import os
import importlib
import inspect
import argparse
import time
import json
import torch
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from PIL import Image, ImageDraw
from diffusers.utils import load_image, is_accelerate_available, is_accelerate_version

# Required for onediffx/nexfort
from onediffx import compile_pipe, quantize_pipe

# --- Argument Parsing ---
def parse_args():
    parser = argparse.ArgumentParser(description="Generate images using diffusion models (T2I/I2I).")
    parser.add_argument("--model", type=str, default=DEFAULT_MODEL, help="Hugging Face model ID or local path.")
    parser.add_argument("--variant", type=str, default=DEFAULT_VARIANT, help="Model variant (e.g., 'fp16').")
    parser.add_argument("--custom-pipeline", type=str, default=DEFAULT_CUSTOM_PIPELINE, help="Custom pipeline class path.")
    parser.add_argument("--scheduler", type=str, default=DEFAULT_SCHEDULER, help="Scheduler name (e.g., 'EulerAncestralDiscreteScheduler', 'DPMSolverMultistepScheduler', 'none').")
    parser.add_argument("--lora", type=str, default=DEFAULT_LORA, help="Path or Hub ID for LoRA weights.")
    parser.add_argument("--controlnet", type=str, default=DEFAULT_CONTROLNET, help="Path or Hub ID for ControlNet model.")
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Number of inference steps.")
    parser.add_argument("--prompt", type=str, default=DEFAULT_PROMPT, help="Text prompt for generation/modification.")
    parser.add_argument("--negative-prompt", type=str, default=DEFAULT_NEGATIVE_PROMPT, help="Negative prompt.")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed for generation. Set to None for random.")
    parser.add_argument("--warmups", type=int, default=DEFAULT_WARMUPS, help="Number of warmup runs before timing.")
    parser.add_argument("--batch", type=int, default=DEFAULT_BATCH, help="Number of images per prompt (batch size).")
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT, help="Image height in pixels. Defaults to model's optimal size.")
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH, help="Image width in pixels. Defaults to model's optimal size.")
    parser.add_argument("--cache_interval", type=int, default=DEFAULT_CACHE_INTERVAL, help="DeepCache cache interval.")
    parser.add_argument("--cache_layer_id", type=int, default=DEFAULT_CACHE_LAYER_ID, help="DeepCache cache layer ID.")
    parser.add_argument("--cache_block_id", type=int, default=DEFAULT_CACHE_BLOCK_ID, help="DeepCache cache block ID.")
    parser.add_argument("--extra-call-kwargs", type=str, default=DEFAULT_EXTRA_CALL_KWARGS, help="JSON string of extra kwargs for the pipeline call (e.g., '{\"strength\": 0.8, \"guidance_scale\": 7.0}').")
    parser.add_argument("--input-image", type=str, default=DEFAULT_INPUT_IMAGE, help="Path or URL to the input image. If provided, enables Image-to-Image mode.")
    parser.add_argument("--control-image", type=str, default=DEFAULT_CONTROL_IMAGE, help="Path or URL to the ControlNet conditioning image.")
    parser.add_argument("--output-image", type=str, default=DEFAULT_OUTPUT_IMAGE, help="Path to save the generated image.")
    parser.add_argument("--throughput", action="store_true", help="Run throughput analysis.")
    parser.add_argument("--deepcache", action="store_true", help="Use DeepCache optimization (if supported by the model/pipeline).")
    parser.add_argument(
        "--task",
        type=str,
        default=DEFAULT_TASK,
        choices=["auto", "text2image", "image2image", "instructpix2pix"],
        help="Specify the task type. 'auto' detects based on --input-image. Use 'instructpix2pix' for that specific pipeline."
    )
    parser.add_argument(
        "--compiler",
        type=str,
        default=DEFAULT_COMPILER,
        choices=["none", "oneflow", "nexfort", "compile", "compile-max-autotune"],
        help="Compiler backend to use for optimization."
    )
    parser.add_argument(
        "--compiler-config",
        type=str,
        default=DEFAULT_COMPILER_CONFIG,
        help="JSON string for compiler configuration options."
    )
    parser.add_argument(
        "--run_multiple_resolutions",
        action="store_true", # Simplified to a flag
        help="Run tests with multiple common resolutions after the main generation."
    )
    parser.add_argument("--quantize", action="store_true", help="Enable quantization (currently requires --compiler nexfort).")
    parser.add_argument(
        "--quantize-config",
        type=str,
        default=DEFAULT_QUANTIZE_CONFIG,
         help="JSON string for quantization configuration."
    )
    parser.add_argument("--quant-submodules-config-path", type=str, default=None, help="Path to quantization submodules config file (for advanced nexfort quantization).")
    return parser.parse_args()

# --- Utility Functions ---

def load_pipe(
    pipeline_cls,
    model_name,
    variant=None,
    dtype=torch.float16,
    device="cuda",
    custom_pipeline=None,
    scheduler=None,
    lora=None,
    controlnet=None,
):
    """Loads the diffusion pipeline with optional components."""
    extra_kwargs = {}
    if custom_pipeline is not None:
        extra_kwargs["custom_pipeline"] = custom_pipeline
    if variant is not None:
        extra_kwargs["variant"] = variant
    if dtype is not None:
        extra_kwargs["torch_dtype"] = dtype

    # Handle ControlNet loading
    if controlnet is not None:
        from diffusers import ControlNetModel
        try:
            controlnet_model = ControlNetModel.from_pretrained(
                controlnet,
                torch_dtype=dtype,
            )
            extra_kwargs["controlnet"] = controlnet_model
            print(f"Successfully loaded ControlNet: {controlnet}")
        except Exception as e:
            print(f"Warning: Failed to load ControlNet '{controlnet}'. Error: {e}")
            print("Proceeding without ControlNet.")
            controlnet = None # Ensure controlnet is None if loading failed

    # Handle pre-quantized models (currently onediff specific)
    if os.path.exists(os.path.join(model_name, "calibrate_info.txt")):
         # Check if QuantPipeline is available before importing
        try:
            from onediff.quantization import QuantPipeline
            print(f"Found quantization info. Loading quantized model: {model_name}")
            pipe = QuantPipeline.from_quantized(pipeline_cls, model_name, **extra_kwargs)
        except ImportError:
            print("Warning: `onediff.quantization.QuantPipeline` not found. Install `onediff` for quantized model support.")
            print("Loading standard pipeline instead.")
            pipe = pipeline_cls.from_pretrained(model_name, **extra_kwargs)
        except Exception as e:
             print(f"Error loading quantized pipeline: {e}. Loading standard pipeline.")
             pipe = pipeline_cls.from_pretrained(model_name, **extra_kwargs)
    else:
        pipe = pipeline_cls.from_pretrained(model_name, **extra_kwargs)

    # Set Scheduler
    if scheduler is not None and scheduler.lower() != "none":
        try:
            scheduler_cls = getattr(importlib.import_module("diffusers"), scheduler)
            pipe.scheduler = scheduler_cls.from_config(pipe.scheduler.config)
            print(f"Using scheduler: {scheduler}")
        except (ImportError, AttributeError, Exception) as e:
            print(f"Warning: Failed to load or set scheduler '{scheduler}'. Using default. Error: {e}")

    # Load LoRA weights
    if lora is not None:
        try:
            print(f"Loading LoRA weights from: {lora}")
            pipe.load_lora_weights(lora)
            # Optionally fuse LoRA - check diffusers version compatibility if issues arise
            if hasattr(pipe, 'fuse_lora'):
                 print("Fusing LoRA weights.")
                 pipe.fuse_lora()
            else:
                 print("Warning: `pipe.fuse_lora()` not found. Skipping fusion (may require newer diffusers version).")
        except Exception as e:
            print(f"Warning: Failed to load or fuse LoRA weights from '{lora}'. Error: {e}")

    # Disable Safety Checker if present
    if hasattr(pipe, "safety_checker"):
        pipe.safety_checker = None
        print("Safety checker disabled.")

    # Move to device
    if device is not None:
        pipe.to(torch.device(device))
        print(f"Pipeline moved to device: {device}")

    return pipe

class IterationProfiler:
    """Profiles iterations per second during pipeline steps."""
    def __init__(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0
        self.enabled = True # Flag to enable/disable profiling easily

    def get_iter_per_sec(self):
        if not self.enabled or self.begin is None or self.end is None or self.num_iterations == 0:
            return None
        try:
            self.end.synchronize() # Ensure timing is accurate
            dur = self.begin.elapsed_time(self.end) # Time in ms
            if dur == 0: return float('inf') # Avoid division by zero
            return self.num_iterations / dur * 1000.0 # Iterations per second
        except Exception as e:
            print(f"Warning: Error during iteration profiling: {e}")
            return None

    def callback_on_step_end(self, pipe, i, t, callback_kwargs={}):
        if not self.enabled:
            return callback_kwargs
        if torch.cuda.is_available():
            if self.begin is None:
                # Start timing on the first step
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.begin = event
                self.num_iterations = 0 # Reset count at start
            else:
                # Record end event on subsequent steps
                event = torch.cuda.Event(enable_timing=True)
                event.record()
                self.end = event
                self.num_iterations += 1 # Increment count *after* the first step completes
        # Pass through callback_kwargs
        return callback_kwargs

    def reset(self):
        self.begin = None
        self.end = None
        self.num_iterations = 0

    def set_enabled(self, enabled=True):
        self.enabled = enabled
        if not enabled:
            self.reset()

# --- Throughput Analysis Functions ---

def calculate_inference_time_and_throughput(pipe, kwarg_inputs, n_steps, profiler):
    """Calculates inference time and throughput for a given number of steps."""
    kwarg_inputs_step = kwarg_inputs.copy()
    kwarg_inputs_step["num_inference_steps"] = n_steps

    # Reset and enable profiler for this run
    profiler.reset()
    profiler.set_enabled(True)

    start_time = time.time()
    # Use dummy generator for throughput test consistency if seed is None originally
    if kwarg_inputs_step.get("generator") is None:
         kwarg_inputs_step["generator"] = torch.Generator(device="cuda").manual_seed(DEFAULT_SEED or 0)

    _ = pipe(**kwarg_inputs_step) # Run inference
    torch.cuda.synchronize() # Ensure completion
    end_time = time.time()

    inference_time = end_time - start_time
    # Use profiler's calculation for it/s based on GPU events
    iter_per_sec = profiler.get_iter_per_sec()
    # Fallback: calculate based on wall time if profiler failed or steps < 2
    if iter_per_sec is None and inference_time > 0 and n_steps > 0 :
        steps_per_sec_wall = n_steps / inference_time
    else:
        steps_per_sec_wall = iter_per_sec if iter_per_sec is not None else 0

    # Disable profiler after use
    profiler.set_enabled(False)

    return inference_time, steps_per_sec_wall


def generate_data_and_fit_model(pipe, base_kwarg_inputs, steps_range, profiler):
    """Generates throughput data across a range of steps and fits a linear model."""
    print("\n--- Starting Throughput Analysis ---")
    data = {"steps": [], "inference_time": [], "throughput": []}
    height = base_kwarg_inputs.get('height', 512)
    width = base_kwarg_inputs.get('width', 512)

    for n_steps in steps_range:
        if n_steps <= 0: continue # Skip invalid step counts
        print(f"Testing {n_steps} steps...")
        try:
            inference_time, throughput = calculate_inference_time_and_throughput(
                pipe, base_kwarg_inputs, n_steps, profiler
            )
            data["steps"].append(n_steps)
            data["inference_time"].append(inference_time)
            # Store throughput (steps/sec)
            data["throughput"].append(throughput)
            print(
                f"  Steps: {n_steps}, Inference Time: {inference_time:.3f}s, Throughput: {throughput:.3f} steps/s"
            )
        except Exception as e:
            print(f"  Error during {n_steps} steps run: {e}")
            # Optionally break or continue
            # break
            continue
        # Short sleep to allow GPU cool-down if needed, can be removed
        # time.sleep(0.5)


    if not data["steps"] or len(data["steps"]) < 2 :
        print("Insufficient data points for throughput modeling.")
        return None, None

    df = pd.DataFrame(data)

    # Calculate Average Throughput from collected data
    # Exclude potential outliers (e.g., first few runs if warmup wasn't enough, or very low step counts)
    # Simple approach: exclude first point or use median/trimmed mean
    valid_throughputs = [t for t in data["throughput"] if t > 0 and np.isfinite(t)]
    if not valid_throughputs:
         print("No valid throughput measurements recorded.")
         average_throughput = 0
    else:
        average_throughput = np.mean(valid_throughputs) # Or np.median(valid_throughputs)
        print(f"\nAverage Measured Throughput: {average_throughput:.3f} steps/s")


    # Fit linear model: time = slope * steps + intercept
    # Requires at least 2 data points
    try:
        coefficients = np.polyfit(df["steps"], df["inference_time"], 1)
        slope = coefficients[0]
        intercept = coefficients[1]
        print(f"Linear Model Fit: Time = {slope:.4f} * Steps + {intercept:.4f}")

        # Estimate throughput based on the slope (time per step)
        if slope > 1e-9: # Avoid division by zero or near-zero
            throughput_from_slope = 1.0 / slope
            print(f"Throughput estimated from slope (ignoring base cost): {throughput_from_slope:.3f} steps/s")
        else:
            print("Slope is too small to estimate throughput reliably.")
            throughput_from_slope = None

    except np.linalg.LinAlgError as e:
        print(f"Could not fit linear model to data: {e}")
        coefficients = None
        throughput_from_slope = None


    print("--- Throughput Analysis Complete ---")
    return data, coefficients


def plot_data_and_model(data, coefficients):
    """Plots the inference time vs steps and the fitted linear model."""
    if data is None or not data["steps"]:
        print("No data to plot.")
        return

    plt.figure(figsize=(10, 6))
    plt.scatter(data["steps"], data["inference_time"], color="blue", label="Measured Data")

    if coefficients is not None and len(coefficients) == 2:
        steps_line = np.array(data["steps"])
        time_line = np.polyval(coefficients, steps_line)
        plt.plot(steps_line, time_line, color="red", label=f"Fit: Time = {coefficients[0]:.4f}*Steps + {coefficients[1]:.4f}")
        plt.legend()

    plt.title("Inference Time vs. Number of Steps")
    plt.xlabel("Number of Inference Steps")
    plt.ylabel("Inference Time (seconds)")
    plt.grid(True)
    plt.tight_layout()

    # Save or show the plot
    plot_filename = "throughput_analysis.png"
    try:
        plt.savefig(plot_filename)
        print(f"Throughput plot saved to {plot_filename}")
        # plt.show() # Uncomment to display interactively if not in a headless environment
    except Exception as e:
        print(f"Error saving/showing plot: {e}")

# --- Main Execution ---
def main():
    args = parse_args()

    # --- Determine Task and Pipeline Class ---
    effective_task = args.task
    if effective_task == "auto":
        if args.input_image is not None:
            effective_task = "image2image"
            print("Detected Image-to-Image task (input image provided).")
        else:
            effective_task = "text2image"
            print("Detected Text-to-Image task (no input image provided).")

    pipeline_cls = None
    if effective_task == "text2image":
        if args.deepcache:
             try:
                 # Requires onediffx.deep_cache module
                 from onediffx.deep_cache import StableDiffusionXLPipeline as PipelineForT2IDeepCache
                 from onediffx.deep_cache import StableDiffusionPipeline as SD15PipelineForT2IDeepCache # Example for SD 1.5
                 # Basic check if model name implies SDXL
                 if "xl" in args.model.lower():
                     pipeline_cls = PipelineForT2IDeepCache
                     print("Using DeepCache SDXL Text-to-Image pipeline.")
                 else:
                     # Assuming SD 1.5/2.1 for non-XL, adjust if needed
                     pipeline_cls = SD15PipelineForT2IDeepCache
                     print("Using DeepCache Stable Diffusion Text-to-Image pipeline.")

             except ImportError:
                 print("Warning: DeepCache pipelines not found in onediffx. Falling back to standard AutoPipeline.")
                 from diffusers import AutoPipelineForText2Image
                 pipeline_cls = AutoPipelineForText2Image
                 args.deepcache = False # Disable deepcache if import failed
        else:
            from diffusers import AutoPipelineForText2Image
            pipeline_cls = AutoPipelineForText2Image
            print("Using AutoPipelineForText2Image.")

    elif effective_task == "image2image":
        # For general I2I, AutoPipelineForImage2Image is suitable
        from diffusers import AutoPipelineForImage2Image
        pipeline_cls = AutoPipelineForImage2Image
        print("Using AutoPipelineForImage2Image.")

    elif effective_task == "instructpix2pix":
        if args.input_image is None:
            raise ValueError("--input-image is required for the 'instructpix2pix' task.")
        # Check if the specified model is likely InstructPix2Pix
        if "instruct-pix2pix" not in args.model.lower():
             print(f"Warning: Model '{args.model}' might not be an InstructPix2Pix model, but task='instructpix2pix' was specified.")
        from diffusers import StableDiffusionInstructPix2PixPipeline
        pipeline_cls = StableDiffusionInstructPix2PixPipeline
        print("Using StableDiffusionInstructPix2PixPipeline.")
        # InstructPix2Pix often uses specific default prompts
        if args.prompt == DEFAULT_PROMPT: # If user didn't override prompt
            args.prompt = "apply the instruction to the image" # More suitable default for instruct-pix2pix
            print(f"Using InstructPix2Pix specific default prompt: '{args.prompt}'")

    else:
        # This case should not be reached due to argparse choices
         raise ValueError(f"Invalid task specified: {args.task}")

    if pipeline_cls is None:
         raise RuntimeError("Could not determine the pipeline class. Check task and model.")


    # --- Load Pipeline ---
    print(f"\nLoading model: {args.model}")
    dtype = torch.float16 if torch.cuda.is_available() else torch.float32 # Use float16 on CUDA by default
    pipe = load_pipe(
        pipeline_cls,
        args.model,
        variant=args.variant,
        custom_pipeline=args.custom_pipeline,
        scheduler=args.scheduler,
        lora=args.lora,
        controlnet=args.controlnet,
        dtype=dtype,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # --- Determine Optimal Height/Width ---
    # Use provided H/W if set, otherwise try to infer from model
    height = args.height
    width = args.width
    if height is None or width is None:
        try:
            model_height = pipe.unet.config.sample_size * pipe.vae_scale_factor
            model_width = pipe.unet.config.sample_size * pipe.vae_scale_factor
            if height is None: height = model_height
            if width is None: width = model_width
            print(f"Auto-detected resolution: {height}x{width}")
        except AttributeError:
            # Fallback if detection fails (e.g., non-standard pipeline structure)
            fallback_res = 512 if "xl" not in args.model.lower() else 1024
            if height is None: height = fallback_res
            if width is None: width = fallback_res
            print(f"Warning: Could not auto-detect resolution. Using default: {height}x{width}")

    # Ensure height and width are multiples of VAE scale factor (usually 8)
    vae_scale_factor = getattr(pipe, "vae_scale_factor", 8)
    height = (height // vae_scale_factor) * vae_scale_factor
    width = (width // vae_scale_factor) * vae_scale_factor
    if args.height != height or args.width != width:
         print(f"Adjusted resolution to be multiples of {vae_scale_factor}: {height}x{width}")


    # --- Apply Compiler/Quantization ---
    compiled = False
    if args.compiler != "none" and torch.cuda.is_available():
        print(f"\nApplying compiler: {args.compiler}")
        if args.compiler == "oneflow":
            # Requires oneflow and onediff to be installed
            try:
                import oneflow # Check if oneflow is installed
                pipe = compile_pipe(pipe, backend="oneflow") # Assumes compile_pipe handles backend selection
                compiled = True
                print("Oneflow backend via compile_pipe is active.")
            except ImportError:
                print("Warning: OneFlow not installed. Skipping OneFlow compilation.")
            except Exception as e:
                print(f"Error during OneFlow compilation: {e}. Proceeding without compilation.")

        elif args.compiler == "nexfort":
            # Requires nexfort, torchao, onediffx
            try:
                quantize_options = {}
                if args.quantize:
                    print("Applying Nexfort quantization...")
                    if args.quantize_config:
                        try:
                            quantize_options = json.loads(args.quantize_config)
                            print(f"Using custom quantize config: {quantize_options}")
                        except json.JSONDecodeError:
                            print(f"Warning: Invalid JSON in --quantize-config. Using default.")
                            quantize_options = {"quant_type": "fp8_e4m3_e4m3_dynamic"} # Default FP8
                    else:
                         quantize_options = {"quant_type": "fp8_e4m3_e4m3_dynamic"} # Default FP8
                         print(f"Using default quantize config: {quantize_options}")

                    if args.quant_submodules_config_path:
                         print(f"Using quant submodules config: {args.quant_submodules_config_path}")
                         pipe = quantize_pipe(
                             pipe,
                             quant_submodules_config_path=args.quant_submodules_config_path,
                             ignores=[], # Example: Add submodules to ignore if needed
                              **quantize_options
                         )
                    else:
                         pipe = quantize_pipe(pipe, ignores=[], **quantize_options)
                    print("Quantization applied.")


                compiler_options = {}
                if args.compiler_config:
                    try:
                        compiler_options = json.loads(args.compiler_config)
                        print(f"Using custom compiler config: {compiler_options}")
                    except json.JSONDecodeError:
                         print(f"Warning: Invalid JSON in --compiler-config. Using default.")
                         # Safe default options string
                         compiler_options = {"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"}
                else:
                    # Safe default options string
                    compiler_options = {"mode": "max-optimize:max-autotune:freezing", "memory_format": "channels_last"}
                    print(f"Using default compiler config: {compiler_options}")

                # Apply compilation
                pipe = compile_pipe(
                    pipe,
                    backend="nexfort",
                    options=compiler_options,
                    fuse_qkv_projections=True # Generally safe and beneficial
                )
                compiled = True
                print("Nexfort backend is active.")

            except ImportError as e:
                 print(f"Warning: Missing dependencies for nexfort ({e}). Skipping nexfort compilation.")
            except Exception as e:
                 print(f"Error during Nexfort setup: {e}. Proceeding without nexfort.")


        elif args.compiler in ("compile", "compile-max-autotune"):
             # Uses torch.compile
            mode = "max-autotune" if args.compiler == "compile-max-autotune" else None
            print(f"Applying torch.compile (mode: {mode or 'default'})...")
            # Compile relevant components
            compiled_components = []
            for component_name in ["unet", "vae", "transformer", "controlnet"]:
                 if hasattr(pipe, component_name) and getattr(pipe, component_name) is not None:
                     try:
                         print(f"Compiling {component_name}...")
                         setattr(pipe, component_name, torch.compile(getattr(pipe, component_name), mode=mode))
                         compiled_components.append(component_name)
                     except Exception as e:
                         print(f"Warning: Failed to compile {component_name}. Error: {e}")

            if compiled_components:
                 print(f"Successfully compiled: {', '.join(compiled_components)}")
                 compiled = True
            else:
                 print("No components were compiled with torch.compile.")

        else:
             # Should not happen due to argparse choices
             print(f"Warning: Unknown compiler '{args.compiler}' requested. Running in eager mode.")
    elif args.compiler != "none":
        print("CUDA not available, skipping compilation.")


    # --- Load Images (Input and Control) ---
    input_image = None
    if args.input_image:
        print(f"Loading input image: {args.input_image}")
        try:
            input_image = load_image(args.input_image)
            input_image = input_image.resize((width, height), Image.LANCZOS)
            print(f"Input image resized to {width}x{height}")
        except Exception as e:
            print(f"Error loading or resizing input image: {e}. Cannot perform Image-to-Image task.")
            return # Exit if I2I is required but image loading fails

    control_image = None
    if args.control_image:
        if args.controlnet is None:
             print("Warning: --control-image provided but no --controlnet model specified. Control image will be ignored.")
        else:
            print(f"Loading control image: {args.control_image}")
            try:
                control_image = load_image(args.control_image)
                control_image = control_image.resize((width, height), Image.LANCZOS)
                print(f"Control image resized to {width}x{height}")
            except Exception as e:
                 print(f"Warning: Error loading or resizing control image: {e}. Proceeding without control image.")
                 control_image = None # Ensure it's None if loading fails
    elif args.controlnet is not None and input_image is not None:
        # If controlnet is specified but no specific control image, use the input image as control
        print("Using input image as control image for ControlNet.")
        control_image = input_image
    elif args.controlnet is not None:
        # ControlNet specified but no control image and no input image (T2I mode)
        # Generate a dummy control image (e.g., blank) or raise error?
        # Let's create a blank white image as a placeholder
        print("Warning: ControlNet specified but no --control-image or --input-image provided.")
        print(f"Creating a blank white control image ({width}x{height}).")
        control_image = Image.new("RGB", (width, height), (255, 255, 255))
        # Or could raise error: raise ValueError("ControlNet requires --control-image or --input-image.")


    # --- Prepare Keyword Arguments for Pipeline Call ---
    def get_kwarg_inputs(current_args, current_height, current_width, current_input_image, current_control_image):
        kwarg_inputs = dict(
            prompt=current_args.prompt,
            negative_prompt=current_args.negative_prompt,
            height=current_height,
            width=current_width,
            num_images_per_prompt=current_args.batch,
            num_inference_steps=current_args.steps, # Ensure steps are included
            generator=(
                None
                if current_args.seed is None
                else torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(current_args.seed)
            ),
        )

        # Add image for I2I tasks
        if effective_task in ["image2image", "instructpix2pix"] and current_input_image is not None:
            kwarg_inputs["image"] = current_input_image
        elif effective_task in ["image2image", "instructpix2pix"]:
             # Should have been caught earlier, but double-check
             raise RuntimeError(f"Input image is required for task '{effective_task}' but is missing.")


        # Add control image if available and ControlNet is loaded
        if current_control_image is not None and hasattr(pipe, 'controlnet') and pipe.controlnet is not None:
             # Some pipelines expect 'control_image', others 'image' if it's the primary input
             # Check signature - this is complex, maybe rely on AutoPipeline or specific pipeline needs
             # Simple approach: Add 'control_image' if ControlNet is present.
             # If it conflicts with 'image', the specific pipeline should handle it or error out.
             kwarg_inputs["control_image"] = current_control_image
             # For T2I + ControlNet, sometimes 'image' needs to be the control image
             if effective_task == "text2image" and "image" not in kwarg_inputs:
                  kwarg_inputs["image"] = current_control_image # Pass control image as 'image' for T2I ControlNet
                  print("Passing control image as 'image' argument for T2I + ControlNet task.")


        # Add DeepCache arguments if enabled
        if current_args.deepcache and effective_task == "text2image": # Currently example DeepCache pipeline is T2I
            # Check if the pipeline actually supports these args (might need more robust check)
            sig = inspect.signature(pipe.__call__)
            if "cache_interval" in sig.parameters:
                kwarg_inputs["cache_interval"] = current_args.cache_interval
                kwarg_inputs["cache_layer_id"] = current_args.cache_layer_id
                kwarg_inputs["cache_block_id"] = current_args.cache_block_id
            else:
                print("Warning: --deepcache specified, but pipeline does not seem to support cache arguments. Disabling.")
                args.deepcache = False # Disable if not supported

        # Add extra keyword arguments from JSON string
        if current_args.extra_call_kwargs:
            try:
                extra_kwargs = json.loads(current_args.extra_call_kwargs)
                # Filter out args already handled explicitly to avoid conflicts
                keys_to_remove = {"prompt", "negative_prompt", "height", "width", "num_images_per_prompt", "generator", "num_inference_steps", "image", "control_image", "cache_interval", "cache_layer_id", "cache_block_id", "callback_on_step_end", "callback"}
                filtered_extra_kwargs = {k: v for k, v in extra_kwargs.items() if k not in keys_to_remove}

                # Check for potential conflicts (e.g., guidance_scale vs EtaDDIM) - let diffusers handle it mostly
                print(f"Adding extra call arguments: {filtered_extra_kwargs}")
                kwarg_inputs.update(filtered_extra_kwargs)
            except json.JSONDecodeError as e:
                print(f"Warning: Invalid JSON in --extra-call-kwargs: {e}. Ignoring extra args.")

        return kwarg_inputs

    # --- Warmup Runs ---
    if args.warmups > 0:
        print("\n=======================================")
        print(f"Begin warmup ({args.warmups} runs)...")
        # Use a temporary profiler for warmup that's disabled
        warmup_profiler = IterationProfiler()
        warmup_profiler.set_enabled(False)
        warmup_kwarg_inputs = get_kwarg_inputs(args, height, width, input_image, control_image)
        # Add dummy callback if needed by signature, but disabled profiler won't use it
        sig = inspect.signature(pipe.__call__)
        if "callback_on_step_end" in sig.parameters:
             warmup_kwarg_inputs["callback_on_step_end"] = warmup_profiler.callback_on_step_end
        elif "callback" in sig.parameters: # Older diffusers convention
             warmup_kwarg_inputs["callback"] = warmup_profiler.callback_on_step_end

        start_warmup_time = time.time()
        for i in range(args.warmups):
            # Ensure generator is reset/new for each warmup if seed is None
            current_seed = args.seed if args.seed is not None else int(time.time()) + i
            warmup_kwarg_inputs["generator"] = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(current_seed)
            _ = pipe(**warmup_kwarg_inputs)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        end_warmup_time = time.time()
        print("End warmup")
        print(f"Warmup time: {end_warmup_time - start_warmup_time:.3f}s")
        print("=======================================")
        del warmup_profiler # Clean up

    # --- Timed Inference Run ---
    print("\n=======================================")
    print("Begin timed inference run...")
    iter_profiler = IterationProfiler()
    # Ensure profiler is enabled for the main run
    iter_profiler.set_enabled(torch.cuda.is_available()) # Only enable if CUDA is available for timing events

    kwarg_inputs = get_kwarg_inputs(args, height, width, input_image, control_image)

    # Add the profiling callback
    sig = inspect.signature(pipe.__call__)
    if "callback_on_step_end" in sig.parameters:
        kwarg_inputs["callback_on_step_end"] = iter_profiler.callback_on_step_end
        print("Iteration profiler attached via callback_on_step_end.")
    elif "callback" in sig.parameters: # Older diffusers convention
        kwarg_inputs["callback"] = iter_profiler.callback_on_step_end
        print("Iteration profiler attached via callback.")
    else:
        iter_profiler.set_enabled(False) # Disable profiler if no callback mechanism found
        print("Warning: Pipeline does not support step callbacks. Iteration profiling disabled.")


    # Clear CUDA cache before timed run (optional, might help consistency)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.max_memory_allocated() / (1024**3) # GiB

    start_time = time.time()
    output_images = pipe(**kwarg_inputs).images
    if torch.cuda.is_available():
        torch.cuda.synchronize() # Wait for GPU to finish
    end_time = time.time()

    inference_time = end_time - start_time
    print("Inference complete.")
    print("=======================================")
    print(f"Task Type: {effective_task.upper()}")
    print(f"Model: {args.model}")
    if compiled: print(f"Compiler: {args.compiler}")
    if args.quantize: print("Quantization: Enabled (Nexfort)")
    print(f"Resolution: {height}x{width}")
    print(f"Steps: {args.steps}")
    print(f"Inference Time (Wall Clock): {inference_time:.3f}s")

    # Report Iterations Per Second from profiler
    iter_per_sec = iter_profiler.get_iter_per_sec()
    if iter_per_sec is not None:
        print(f"Iterations Per Second (GPU Profiled): {iter_per_sec:.3f}")
    elif iter_profiler.enabled and args.steps > 1:
         print("Iterations Per Second (GPU Profiled): N/A (profiling error or too few steps)")
    elif inference_time > 0 and args.steps > 0:
        # Fallback to wall clock steps/sec if profiler not available/failed
        wall_steps_per_sec = args.steps / inference_time
        print(f"Steps Per Second (Wall Clock): {wall_steps_per_sec:.3f}")


    # Report Memory Usage
    if torch.cuda.is_available():
        # Use torch.cuda.max_memory_allocated() which tracks peak usage
        cuda_mem_after_used = torch.cuda.max_memory_allocated() / (1024**3) # GiB
        # Reset peak stats for next potential runs if needed
        torch.cuda.reset_peak_memory_stats()
        print(f"Max CUDA Memory Used: {cuda_mem_after_used:.3f} GiB")
    # elif args.compiler == "oneflow": # Specific check for oneflow if needed
    #     try:
    #         import oneflow as flow
    #         # Note: OneFlow's memory reporting might differ
    #         cuda_mem_after_used = flow._oneflow_internal.GetCUDAMemoryUsed() / 1024 # KiB? Check unit
    #         print(f"Max used OneFlow CUDA memory : {cuda_mem_after_used:.3f} Units (check OneFlow docs for unit)")
    #     except ImportError:
    #         pass # Oneflow not installed
    print("=======================================")

    # --- Save Output Image ---
    if args.output_image:
        try:
            output_images[0].save(args.output_image)
            print(f"Output image saved to: {args.output_image}")
        except IndexError:
            print("Error: No images generated.")
        except Exception as e:
            print(f"Error saving output image to {args.output_image}: {e}")
    else:
        print("No --output-image path specified. Image not saved.")

    # --- Optional: Run Multiple Resolutions Test ---
    if args.run_multiple_resolutions:
        print("\n--- Testing Multiple Resolutions ---")
        sizes = [1024, 768, 512, 256] # Example sizes
        base_kwarg_inputs = get_kwarg_inputs(args, height, width, input_image, control_image)
        # Remove callbacks for these runs if they were added
        base_kwarg_inputs.pop("callback_on_step_end", None)
        base_kwarg_inputs.pop("callback", None)

        for h in sizes:
            for w in sizes:
                 # Skip if same as original run
                 if h == height and w == width: continue

                 # Adjust resolution, ensuring it's valid (multiple of 8)
                 h_test = (h // vae_scale_factor) * vae_scale_factor
                 w_test = (w // vae_scale_factor) * vae_scale_factor
                 if h_test == 0 or w_test == 0: continue # Skip invalid zero sizes

                 print(f"Running at resolution: {h_test}x{w_test}")
                 current_kwarg_inputs = base_kwarg_inputs.copy()
                 current_kwarg_inputs["height"] = h_test
                 current_kwarg_inputs["width"] = w_test

                 # Need to resize input/control images if they exist for I2I
                 current_input_image_test = None
                 if input_image:
                      try:
                          current_input_image_test = input_image.resize((w_test, h_test), Image.LANCZOS)
                          if effective_task in ["image2image", "instructpix2pix"]:
                               current_kwarg_inputs["image"] = current_input_image_test
                      except Exception as e:
                          print(f"  Warn: Failed to resize input image for {h_test}x{w_test}. Skipping.")
                          continue

                 current_control_image_test = None
                 if control_image:
                      try:
                          current_control_image_test = control_image.resize((w_test, h_test), Image.LANCZOS)
                          if "control_image" in current_kwarg_inputs: # Check if key exists
                               current_kwarg_inputs["control_image"] = current_control_image_test
                          if effective_task == "text2image" and args.controlnet: # T2I+ControlNet case
                              current_kwarg_inputs["image"] = current_control_image_test
                      except Exception as e:
                           print(f"  Warn: Failed to resize control image for {h_test}x{w_test}. Skipping.")
                           continue


                 # Reset generator for consistency if seed is None
                 if args.seed is None:
                     current_kwarg_inputs["generator"] = torch.Generator(device="cuda" if torch.cuda.is_available() else "cpu").manual_seed(DEFAULT_SEED or 0) # Use default seed for test runs


                 try:
                     start_res_time = time.time()
                     _ = pipe(**current_kwarg_inputs).images
                     if torch.cuda.is_available(): torch.cuda.synchronize()
                     end_res_time = time.time()
                     print(f"  Inference time: {end_res_time - start_res_time:.3f} seconds")
                 except Exception as e:
                     print(f"  Error during {h_test}x{w_test} run: {e}")
                 # Optional: small delay
                 # time.sleep(0.5)
        print("--- Multi-resolution Testing Complete ---")


    # --- Optional: Throughput Analysis ---
    if args.throughput:
        # Use a range starting from a few steps up to maybe slightly more than default
        steps_range = range(5, max(55, args.steps + 15), 5) # e.g., 5, 10, 15... up to 50 or more
        base_kwarg_inputs = get_kwarg_inputs(args, height, width, input_image, control_image)
        # Remove callbacks for throughput runs as we use a dedicated profiler inside
        base_kwarg_inputs.pop("callback_on_step_end", None)
        base_kwarg_inputs.pop("callback", None)

        throughput_data, throughput_coeffs = generate_data_and_fit_model(
            pipe, base_kwarg_inputs, steps_range, iter_profiler # Reuse main profiler object
            )
        if throughput_data:
             plot_data_and_model(throughput_data, throughput_coeffs)

if __name__ == "__main__":
    main()
