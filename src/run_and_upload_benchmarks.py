#!/usr/bin/env python3
import os
import subprocess
import argparse
from typing import List, Optional

def run_command(cmd: List[str], description: str) -> bool:
    """Run a command and return True if successful."""
    print(f"\n{'='*80}")
    print(f"Running: {description}")
    print(f"{'='*80}")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {e}")
        return False
    except Exception as e:
        print(f"Unexpected error: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run benchmarks and upload to Hugging Face')
    parser.add_argument('--local', action='store_true', help='Run in local testing mode with limited prompts')
    parser.add_argument('--models', nargs='+', help='Specific models to benchmark (by model_name)')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    parser.add_argument('--dataset', type=str, default='ohdoking/energy_consumption_by_model_and_gpu',
                      help='Hugging Face dataset name')
    args = parser.parse_args()
    
    # Build benchmark command
    benchmark_cmd = ["poetry", "run", "python", "src/run_benchmarks.py"]
    if args.local:
        benchmark_cmd.append("--local")
    if args.models:
        benchmark_cmd.extend(["--models"] + args.models)
    
    # Run benchmarks
    if not run_command(benchmark_cmd, "Running benchmarks"):
        print("Benchmarking failed. Aborting upload.")
        return 1
    
    # Build upload command
    upload_cmd = [
        "poetry", "run", "python", "src/upload_benchmarks_to_hf.py",
        "--token", args.token,
        "--dataset", args.dataset
    ]
    
    # Upload to Hugging Face
    if not run_command(upload_cmd, "Uploading to Hugging Face"):
        print("Upload failed.")
        return 1
    
    print("\nBenchmark and upload completed successfully!")
    print(f"Dataset available at: https://huggingface.co/datasets/{args.dataset}")
    return 0

if __name__ == "__main__":
    exit(main()) 