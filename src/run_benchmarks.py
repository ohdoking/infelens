#!/usr/bin/env python3
import os
import json
import subprocess
import argparse
from typing import List, Dict
from dataclasses import dataclass
from datetime import datetime

@dataclass
class LLMModel:
    model_name: str
    huggingface_model: str
    num_params: str
    hidden_size: int
    num_layers: int
    vocab_size: int
    seq_length: int
    model_type: str

@dataclass
class BenchmarkResult:
    model_name: str
    huggingface_model: str
    metrics_file: str
    timestamp: str
    hardware_info: Dict
    total_prompts: int
    summary: Dict
    prompts: List[Dict]

def load_models(local: bool = False, validate: bool = False) -> List[LLMModel]:
    """Load model configurations from JSON file.
    
    Args:
        local: If True, use llm_models.local.json
        validate: If True, use validation_llm_models.json
        If both are False, use llm_models.json
    """
    if local and validate:
        print("Error: Cannot use both --local and --validate flags together")
        return []
        
    if local:
        model_file = 'llm_models.local.json'
    elif validate:
        model_file = 'validation_llm_models.json'
    else:
        model_file = 'llm_models.json'
    
    try:
        with open(os.path.join('data', model_file), 'r') as f:
            models_data = json.load(f)
        print(f"Loaded {len(models_data)} models from {model_file}")
        return [LLMModel(**model) for model in models_data]
    except Exception as e:
        print(f"Error loading models from {model_file}: {e}")
        return []

def parse_args():
    parser = argparse.ArgumentParser(description='Run LLM benchmarks across multiple models')
    parser.add_argument('--local', action='store_true', help='Run in local testing mode with limited prompts')
    parser.add_argument('--models', nargs='+', help='Specific models to benchmark (by model_name, default: all in llm_models.json)')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file for merged results')
    parser.add_argument('--validate', action='store_true', help='Use validation models instead of full model list')
    return parser.parse_args()

def ensure_output_dir():
    """Ensure the output directory exists, create it if it doesn't."""
    output_dir = "output"
    if not os.path.exists(output_dir):
        try:
            os.makedirs(output_dir)
            print(f"Created output directory: {output_dir}")
        except Exception as e:
            print(f"Error creating output directory: {e}")
            raise

def run_benchmark(model: LLMModel, local: bool = False) -> str:
    """Run the benchmark script for a single model and return the output metrics file path."""
    print(f"\n{'='*80}")
    print(f"Running benchmark for model: {model.model_name} ({model.huggingface_model})")
    print(f"Model details: {model.num_params} parameters, {model.num_layers} layers")
    print(f"{'='*80}")
    
    # Ensure output directory exists before running benchmark
    ensure_output_dir()
    
    cmd = ["poetry", "run", "python", "src/check_energy_consumption_script.py", "--model", model.huggingface_model]
    if local:
        cmd.append("--local")
    
    try:
        subprocess.run(cmd, check=True)
        
        # Find the most recent metrics file for this model
        output_dir = "output"
        model_files = [f for f in os.listdir(output_dir) 
                      if f.startswith(f"prompt_metrics_") and f.endswith(".json")]
        
        if not model_files:
            raise FileNotFoundError(f"No metrics file found for model {model.model_name}")
        
        # Sort by modification time and get the most recent
        latest_file = max(model_files, key=lambda x: os.path.getmtime(os.path.join(output_dir, x)))
        return os.path.join(output_dir, latest_file)
    
    except subprocess.CalledProcessError as e:
        print(f"Error running benchmark for model {model.model_name}: {e}")
        return None
    except Exception as e:
        print(f"Unexpected error for model {model.model_name}: {e}")
        return None

def load_metrics(file_path: str, model: LLMModel) -> BenchmarkResult:
    """Load metrics from a JSON file and return a BenchmarkResult object."""
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    # Extract timestamp from filename
    timestamp = os.path.basename(file_path).split('_')[-1].replace('.json', '')
    
    return BenchmarkResult(
        model_name=model.model_name,
        huggingface_model=model.huggingface_model,
        metrics_file=file_path,
        timestamp=timestamp,
        hardware_info=data['hardware_info'],
        total_prompts=data['total_prompts'],
        summary=data['summary'],
        prompts=data['prompts']
    )

def merge_results(results: List[BenchmarkResult], output_file: str, local: bool = False, validate: bool = False):
    """Merge all benchmark results into a single JSON file."""
    # Find the original model information for each benchmark result
    all_models = load_models(local=local, validate=validate)
    model_info_map = {m.huggingface_model: m for m in all_models}
    
    merged_data = {
        "benchmark_timestamp": datetime.now().strftime("%Y%m%d_%H%M%S"),
        "models": [
            {
                "model_name": r.model_name,
                "huggingface_model": r.huggingface_model,
                "model_info": {
                    "num_params": model_info_map[r.huggingface_model].num_params,
                    "hidden_size": model_info_map[r.huggingface_model].hidden_size,
                    "num_layers": model_info_map[r.huggingface_model].num_layers,
                    "vocab_size": model_info_map[r.huggingface_model].vocab_size,
                    "seq_length": model_info_map[r.huggingface_model].seq_length,
                    "model_type": model_info_map[r.huggingface_model].model_type
                },
                "timestamp": r.timestamp,
                "hardware_info": r.hardware_info,
                "total_prompts": r.total_prompts,
                "summary": r.summary,
                "prompts": r.prompts
            }
            for r in results
        ],
        "comparison_summary": {
            "total_models": len(results),
            "models_compared": [
                {
                    "model_name": r.model_name,
                    "huggingface_model": r.huggingface_model,
                    "num_params": model_info_map[r.huggingface_model].num_params,
                    "model_type": model_info_map[r.huggingface_model].model_type,
                    "metrics": {
                        "average_runtime": r.summary['average_runtime'],
                        "average_energy": r.summary['average_energy'],
                        "average_co2": r.summary['average_co2']
                    }
                }
                for r in results
            ],
            "average_runtime": sum(r.summary['average_runtime'] for r in results) / len(results),
            "average_energy": sum(r.summary['average_energy'] for r in results) / len(results),
            "average_co2": sum(r.summary['average_co2'] for r in results) / len(results)
        }
    }
    
    # Save timestamped version
    with open(output_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"\nMerged results saved to: {output_file}")
    
    # Save latest version
    latest_file = os.path.join(os.path.dirname(output_file), "merged_benchmarks_latest.json")
    with open(latest_file, 'w') as f:
        json.dump(merged_data, f, indent=2)
    print(f"Latest results saved to: {latest_file}")

def main():
    args = parse_args()
    
    # Ensure output directory exists at the start
    try:
        ensure_output_dir()
    except Exception as e:
        print("Failed to create output directory. Cannot proceed with benchmarks.")
        return
    
    # Load models from appropriate config file based on flags
    all_models = load_models(local=args.local, validate=args.validate)
    if not all_models:
        print("No models loaded! Cannot proceed with benchmarks.")
        return
    
    # Filter models if specific ones are requested
    if args.models:
        models_to_run = [m for m in all_models if m.model_name in args.models]
        if not models_to_run:
            print(f"No models found matching: {args.models}")
            print("Available models:")
            for m in all_models:
                print(f"  - {m.model_name}")
            return
    else:
        models_to_run = all_models
    
    # Run benchmarks for each model
    results = []
    for model in models_to_run:
        metrics_file = run_benchmark(model, args.local)
        if metrics_file:
            try:
                result = load_metrics(metrics_file, model)
                results.append(result)
                print(f"Successfully loaded metrics for {model.model_name}")
            except Exception as e:
                print(f"Error loading metrics for {model.model_name}: {e}")
    
    if not results:
        print("No successful benchmark results to merge!")
        return
    
    # Determine output filename
    if args.output is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.output = os.path.join("output", f"merged_benchmarks_{timestamp}.json")
    else:
        args.output = os.path.join("output", args.output)
    
    # Merge and save results (both timestamped and latest versions)
    merge_results(results, args.output, args.local, args.validate)
    
    # Print summary
    print("\nBenchmark Summary:")
    print(f"Total models benchmarked: {len(results)}")
    print("Models:")
    for r in results:
        model_info = next((m for m in all_models if m.huggingface_model == r.huggingface_model), None)
        print(f"  - {r.model_name} ({r.huggingface_model})")
        print(f"    Parameters: {model_info.num_params}, Layers: {model_info.num_layers}")
        print(f"    Average runtime: {r.summary['average_runtime']:.2f} seconds")
        print(f"    Average energy: {r.summary['average_energy']:.2f} Joules")
        print(f"    Average CO2: {r.summary['average_co2']:.6f} kg CO2eq")

if __name__ == "__main__":
    main() 