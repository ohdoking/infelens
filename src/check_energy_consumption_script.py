# Import necessary libraries
import time
import os
import json
import argparse
from dataclasses import dataclass
from typing import List, Dict
import torch
from transformers import pipeline, set_seed, AutoModelForCausalLM, AutoTokenizer
from codecarbon import EmissionsTracker

# Suppress tokenizers warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Ensure required directories exist
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

@dataclass
class PromptMetrics:
    prompt: str
    runtime: float
    energy_joules: float
    co2_emissions: float
    response: str

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM energy consumption test')
    parser.add_argument('--local', action='store_true', help='Run in local testing mode with limited prompts')
    parser.add_argument('--model', type=str, default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                      help='Model to use for testing')
    parser.add_argument('--output', type=str, default=None,
                      help='Output JSON file for detailed metrics (if not specified, will use GPU name and timestamp)')
    return parser.parse_args()

args = parse_args()

def get_gpu_info() -> Dict:
    """Get information about available GPUs."""
    gpu_info = {
        "gpu_available": torch.cuda.is_available(),
        "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "gpu_devices": [],
        "device_name": "CPU"  # Default to CPU
    }
    
    if gpu_info["gpu_available"]:
        for i in range(gpu_info["gpu_count"]):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_info["gpu_devices"].append({
                "name": gpu_name,
                "memory_total": f"{torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB",
                "memory_free": f"{torch.cuda.memory_reserved(i) / 1024**3:.2f} GB"
            })
            # Use the first GPU's name for the filename
            if i == 0:
                # Clean the GPU name for filename use (replace spaces and special characters)
                gpu_info["device_name"] = gpu_name.replace(" ", "_").replace("(", "").replace(")", "")
    else:
        gpu_info["gpu_devices"].append({
            "name": "CPU",
            "memory_total": "N/A",
            "memory_free": "N/A"
        })
    
    return gpu_info

# Model configuration
MODEL_NAME = args.model
OUTPUT_FILE = "emissions.csv"  # codecarbon will write its output here

# --- Load Prompts from JSON ---
print("Loading prompts from data/sample_prompts.json...")
try:
    with open(os.path.join(DATA_DIR, "sample_prompts.json"), "r") as f:
        data = json.load(f)
        user_prompts = [item["prompt"] for item in data["prompts"]]
        if args.local:
            # For local testing, use only 5 prompts
            user_prompts = user_prompts[:5]
            print("Running in local testing mode with 5 prompts")
        NUM_PROMPTS = len(user_prompts)
        print(f"Successfully loaded {NUM_PROMPTS} prompts from data/sample_prompts.json")
except Exception as e:
    print(f"Error loading prompts from JSON: {e}")
    exit()

# Ensure the output directory for codecarbon exists if not in current directory
if not os.path.exists(os.path.dirname(OUTPUT_FILE)) and os.path.dirname(OUTPUT_FILE):
    os.makedirs(os.path.dirname(OUTPUT_FILE))

print(f"--- Starting LLM Performance Measurement ---")
print(f"Model: {MODEL_NAME}")
print(f"Number of prompts: {NUM_PROMPTS}")
print(f"CodeCarbon output will be saved to: {OUTPUT_FILE}")

# --- Initialize Model and Tokenizer ---
print("\nLoading model and tokenizer...")
try:
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",  # This will automatically use CPU if no GPU is available
        torch_dtype="auto"  # This will automatically choose the appropriate dtype
    )
    
    # Create a text generation pipeline with proper chat template
    generator = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=None,  # No token limit
        device_map="auto",
        do_sample=False,  # Disable sampling for deterministic results
        num_beams=1,  # Use greedy decoding
        temperature=1.0,  # Neutral temperature
        repetition_penalty=1.0  # No repetition penalty
    )
    
    set_seed(42)  # For reproducibility
    print("Model loaded successfully.")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Please ensure you have the required packages installed:")
    print("pip install transformers torch accelerate bitsandbytes")
    exit()

# --- Measure Runtime and Energy Consumption ---
print("\nRunning inference and measuring performance...")

# Initialize single tracker for all prompts
tracker = EmissionsTracker(
    output_file=OUTPUT_FILE,
    project_name="LLM_Benchmark",
    measure_power_secs=1,
    tracking_mode="process"
)

# Initialize list to store metrics for each prompt
prompt_metrics: List[PromptMetrics] = []

try:
    # Start tracking for all prompts
    tracker.start()
    total_start_time = time.perf_counter()
    
    # Process all prompts
    for i, prompt in enumerate(user_prompts):
        if i % max(1, NUM_PROMPTS // 10) == 0:  # Print progress every 10%
            print(f"\nProcessing prompt {i+1}/{NUM_PROMPTS}...")
        
        prompt_start_time = time.perf_counter()
        
        # Format prompt for chat model
        chat_prompt = f"<|im_start|>user\n{prompt}<|im_end|>\n<|im_start|>assistant\n"
        
        # Generate response
        response = generator(chat_prompt, max_new_tokens=None)  # No token limit
        full_response = response[0]['generated_text']
        
        # Extract just the assistant's response and remove all whitespace
        response_text = full_response[len(chat_prompt):]
        response_text = response_text.replace("\t", "").replace("\n", "").replace("\r", "")
        
        # Calculate metrics for this prompt
        prompt_end_time = time.perf_counter()
        prompt_runtime = prompt_end_time - prompt_start_time
        
        # Store metrics (we'll calculate energy and CO2 at the end)
        metrics = PromptMetrics(
            prompt=prompt,
            runtime=prompt_runtime,
            energy_joules=0.0,  # Will be calculated later
            co2_emissions=0.0,  # Will be calculated later
            response=response_text
        )
        prompt_metrics.append(metrics)
        
        # Print prompt-specific results
        print(f"\nResults for Prompt {i+1}:")
        print(f"Prompt: {prompt[:100]}..." if len(prompt) > 100 else f"Prompt: {prompt}")
        print(f"Runtime: {prompt_runtime:.2f} seconds")
        print(f"Response Preview: {response_text[:100]}..." if len(response_text) > 100 else f"Response: {response_text}")
    
    # End tracking for all prompts
    total_end_time = time.perf_counter()
    total_emissions = tracker.stop()
    
    # Calculate total energy consumption
    try:
        with open(OUTPUT_FILE, 'r') as f:
            lines = f.readlines()
            if len(lines) > 1:  # Skip header
                last_line = lines[-1].strip().split(',')
                total_energy_kwh = float(last_line[11])  # energy_consumed is at index 11
                total_energy_joules = total_energy_kwh * 3.6 * 10**6
                
                # Distribute energy consumption proportionally based on runtime
                total_runtime = sum(m.runtime for m in prompt_metrics)
                for metrics in prompt_metrics:
                    metrics.energy_joules = (metrics.runtime / total_runtime) * total_energy_joules
                    metrics.co2_emissions = (metrics.runtime / total_runtime) * float(total_emissions) if total_emissions is not None else 0.0
    except Exception as e:
        print(f"Warning: Could not read energy data: {e}")
    
except Exception as e:
    print(f"Error during benchmark: {e}")
    tracker.stop()  # Make sure to stop the tracker even if there's an error

# Get GPU info early to use in filename
gpu_info = get_gpu_info()

# Determine output filename
if args.output is None:
    # Add timestamp
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    # Create filename based on GPU name and timestamp
    base_filename = f"prompt_metrics_{gpu_info['device_name']}_{timestamp}.json"
    # Ensure the filename is valid
    base_filename = "".join(c for c in base_filename if c.isalnum() or c in ('_', '-', '.'))
    args.output = os.path.join(OUTPUT_DIR, base_filename)
else:
    # If custom output name is provided, still put it in the output directory
    args.output = os.path.join(OUTPUT_DIR, args.output)

# Save detailed metrics to JSON file
try:
    metrics_dict = {
        "model": MODEL_NAME,
        "total_prompts": NUM_PROMPTS,
        "hardware_info": gpu_info,  # Use the already fetched GPU info
        "prompts": [
            {
                "prompt": m.prompt,
                "runtime_seconds": m.runtime,
                "energy_joules": m.energy_joules,
                "co2_emissions_kg": m.co2_emissions,
                "response": m.response
            }
            for m in prompt_metrics
        ],
        "summary": {
            "total_runtime": sum(m.runtime for m in prompt_metrics),
            "total_energy": sum(m.energy_joules for m in prompt_metrics),
            "total_co2": sum(m.co2_emissions for m in prompt_metrics),
            "average_runtime": sum(m.runtime for m in prompt_metrics) / len(prompt_metrics) if prompt_metrics else 0,
            "average_energy": sum(m.energy_joules for m in prompt_metrics) / len(prompt_metrics) if prompt_metrics else 0,
            "average_co2": sum(m.co2_emissions for m in prompt_metrics) / len(prompt_metrics) if prompt_metrics else 0
        }
    }
    
    with open(args.output, 'w') as f:
        json.dump(metrics_dict, f, indent=2)
    print(f"\nDetailed metrics saved to: {args.output}")
    
    # Print hardware info
    print("\nHardware Information:")
    if metrics_dict["hardware_info"]["gpu_available"]:
        print(f"GPU Available: Yes")
        print(f"Number of GPUs: {metrics_dict['hardware_info']['gpu_count']}")
        for i, gpu in enumerate(metrics_dict["hardware_info"]["gpu_devices"]):
            print(f"GPU {i}: {gpu['name']}")
            print(f"  Total Memory: {gpu['memory_total']}")
            print(f"  Free Memory: {gpu['memory_free']}")
    else:
        print("GPU Available: No (Running on CPU)")
    
except Exception as e:
    print(f"Error saving metrics to JSON: {e}")

print("\n--- Script Finished ---")
print(f"Detailed metrics for each prompt have been saved to: {os.path.abspath(args.output)}")


