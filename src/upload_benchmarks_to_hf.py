#!/usr/bin/env python3
import os
import json
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Value, Sequence
from huggingface_hub import HfApi, create_repo
from typing import Dict, List
import argparse

def load_merged_benchmarks(file_path: str) -> Dict:
    """Load merged benchmarks from JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def convert_to_dataframe(benchmarks_data: Dict) -> pd.DataFrame:
    """Convert merged benchmarks data to a pandas DataFrame."""
    rows = []
    
    for model_data in benchmarks_data['models']:
        # Extract model info
        model_info = model_data['model_info']
        hardware_info = model_data['hardware_info']
        summary = model_data['summary']
        
        # Get GPU info safely
        gpu_name = ''
        gpu_memory = ''
        if hardware_info.get('gpu_available') and hardware_info.get('gpu_devices'):
            gpu_name = hardware_info['gpu_devices'][0].get('name', '')
            gpu_memory = hardware_info['gpu_devices'][0].get('memory_total', '')
        
        # Create base row with model and hardware info
        row = {
            'model_name': model_data['model_name'],
            'huggingface_model': model_data['huggingface_model'],
            'num_params': model_info['num_params'],
            'hidden_size': model_info['hidden_size'],
            'num_layers': model_info['num_layers'],
            'vocab_size': model_info['vocab_size'],
            'seq_length': model_info['seq_length'],
            'model_type': model_info['model_type'],
            'timestamp': model_data['timestamp'],
            'hardware_cpu': hardware_info.get('device_name', ''),
            'hardware_gpu': gpu_name,
            'hardware_ram': gpu_memory,
            'total_prompts': model_data['total_prompts'],
            'average_runtime': summary['average_runtime'],
            'average_energy': summary['average_energy'],
            'average_co2': summary['average_co2'],
            'benchmark_timestamp': benchmarks_data['benchmark_timestamp']
        }
        
        # Add per-prompt metrics
        for i, prompt_data in enumerate(model_data['prompts']):
            prompt_row = row.copy()
            # Get the model's response, which is the full generated text minus the prompt
            prompt_text = prompt_data['prompt']
            full_response = prompt_data['response']
            # Remove the prompt from the response if it's included
            response_text = full_response[len(prompt_text):] if full_response.startswith(prompt_text) else full_response
            
            prompt_row.update({
                'prompt_index': i + 1,  # Use 1-based index as prompt identifier
                'prompt_text': prompt_text,
                'prompt_runtime': prompt_data['runtime_seconds'],
                'prompt_energy': prompt_data['energy_joules'],
                'prompt_co2': prompt_data['co2_emissions_kg'],
                'prompt_response': response_text.strip()  # Remove any leading/trailing whitespace
            })
            rows.append(prompt_row)
    
    return pd.DataFrame(rows)

def create_hf_dataset(df: pd.DataFrame) -> Dataset:
    """Create a Hugging Face dataset from the DataFrame with feature descriptions."""
    # Define feature descriptions
    feature_descriptions = {
        # Model information
        'model_name': 'Name of the language model',
        'huggingface_model': 'Hugging Face model identifier',
        'num_params': 'Number of parameters in the model (e.g., "1.1B" for 1.1 billion)',
        'hidden_size': 'Size of the hidden layers in the model',
        'num_layers': 'Number of transformer layers in the model',
        'vocab_size': 'Size of the model\'s vocabulary',
        'seq_length': 'Maximum sequence length the model can process',
        'model_type': 'Type of transformer architecture used',
        
        # Hardware information
        'hardware_cpu': 'CPU device name used for inference',
        'hardware_gpu': 'GPU device name used for inference (if available)',
        'hardware_ram': 'Total RAM available on the system',
        
        # Benchmark metadata
        'timestamp': 'Timestamp of the individual model benchmark run',
        'benchmark_timestamp': 'Timestamp of the complete benchmark run',
        'total_prompts': 'Total number of prompts processed for this model',
        
        # Summary metrics
        'average_runtime': 'Average runtime per prompt in seconds',
        'average_energy': 'Average energy consumption per prompt in Joules',
        'average_co2': 'Average CO2 emissions per prompt in kg CO2eq',
        
        # Per-prompt metrics
        'prompt_index': 'Index of the prompt in the benchmark sequence',
        'prompt_text': 'The input prompt text',
        'prompt_runtime': 'Runtime for this specific prompt in seconds',
        'prompt_energy': 'Energy consumption for this prompt in Joules',
        'prompt_co2': 'CO2 emissions for this prompt in kg CO2eq',
        'prompt_response': 'Model\'s response to the prompt'
    }
    
    # Create features dictionary with proper types
    features_dict = {}
    for name in df.columns:
        if name in feature_descriptions:
            # Determine the correct type
            if df[name].dtype == 'object':
                dtype = 'string'
            elif df[name].dtype == 'int64':
                dtype = 'int64'
            else:
                dtype = 'float64'
            
            # Create feature with type
            features_dict[name] = Value(dtype)
    
    # Create features object
    features = Features(features_dict)
    
    # Create dataset with features
    dataset = Dataset.from_pandas(df, features=features)
    
    # Add feature descriptions after dataset creation
    for name, description in feature_descriptions.items():
        if name in dataset.features:
            dataset.features[name].description = description
    
    # Add dataset description and metadata
    dataset.info.description = """
    This dataset contains energy consumption and performance metrics for various Language Learning Models (LLMs).
    It includes detailed measurements of runtime, energy consumption, and CO2 emissions for each model and prompt.
    The data is collected using CodeCarbon for energy tracking and includes both model specifications and hardware information.
    """
    dataset.info.license = "MIT"
    dataset.info.citation = """
    @software{infelens2024,
        author = {Your Name},
        title = {Infelens: LLM Energy Consumption Measurement Tool},
        year = {2024},
        publisher = {Hugging Face},
        journal = {Hugging Face Hub},
        howpublished = {\\url{https://huggingface.co/datasets/ohdoking/energy_consumption_by_model_and_gpu}}
    }
    """
    
    return DatasetDict({'train': dataset})

def ensure_repo_exists(api: HfApi, dataset_name: str, token: str):
    """Ensure the dataset repository exists."""
    try:
        api.repo_info(repo_id=dataset_name, repo_type="dataset")
    except Exception:
        print(f"Creating dataset repository: {dataset_name}")
        create_repo(
            repo_id=dataset_name,
            repo_type="dataset",
            token=token,
            exist_ok=True
        )

def get_hardware_name(df: pd.DataFrame) -> str:
    """Extract hardware name from the dataset for filename."""
    # Get unique hardware combinations
    hardware_info = df[['hardware_cpu', 'hardware_gpu']].drop_duplicates()
    if len(hardware_info) > 1:
        print("Warning: Multiple hardware configurations found. Using the first one for filename.")
    
    # Get the first hardware configuration
    cpu = hardware_info.iloc[0]['hardware_cpu']
    gpu = hardware_info.iloc[0]['hardware_gpu']
    
    # Use GPU name if available, otherwise use CPU name
    hardware_name = gpu if gpu else cpu
    # Clean the hardware name for filename use
    hardware_name = hardware_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
    return hardware_name

def upload_to_hf(dataset: DatasetDict, dataset_name: str, token: str, df: pd.DataFrame):
    """Upload dataset to Hugging Face with custom filename inside data folder."""
    api = HfApi(token=token)
    
    # Ensure repository exists
    ensure_repo_exists(api, dataset_name, token)
    
    # Get hardware name for filename
    hardware_name = get_hardware_name(df)
    
    # Get benchmark timestamp
    benchmark_timestamp = df['benchmark_timestamp'].iloc[0]
    timestamp_str = benchmark_timestamp.replace(' ', '_').replace(':', '-')
    
    # Create custom filename
    custom_filename = f"benchmark_{timestamp_str}_{hardware_name}.csv"
    
    # Save dataset locally with custom filename
    temp_dir = "temp_dataset"
    os.makedirs(temp_dir, exist_ok=True)
    dataset_path = os.path.join(temp_dir, custom_filename)
    
    # Save the dataset as CSV
    dataset['train'].to_csv(dataset_path, index=False)
    
    # Push to hub inside data folder
    print(f"Uploading to {dataset_name}/data/{custom_filename}...")
    api.upload_file(
        path_or_fileobj=dataset_path,
        path_in_repo=f"data/{custom_filename}",  # Upload to data folder
        repo_id=dataset_name,
        repo_type="dataset",
        commit_message=f"Update benchmark results for {hardware_name}"
    )
    
    # Clean up temporary directory
    import shutil
    shutil.rmtree(temp_dir)

def main():
    parser = argparse.ArgumentParser(description='Convert latest benchmarks to CSV and upload to Hugging Face')
    parser.add_argument('--dataset', type=str, default='ohdoking/energy_consumption_by_model_and_gpu',
                      help='Hugging Face dataset name')
    parser.add_argument('--token', type=str, required=True, help='Hugging Face API token')
    args = parser.parse_args()
    
    try:
        # Use the latest merged benchmarks file
        input_file = os.path.join('output', 'merged_benchmarks_latest.json')
        if not os.path.exists(input_file):
            print(f"Error: Latest benchmarks file not found at {input_file}")
            print("Please run benchmarks first using: poetry run python src/run_benchmarks.py")
            return 1
        
        # Load and convert data
        print(f"Loading latest benchmarks from {input_file}...")
        benchmarks_data = load_merged_benchmarks(input_file)
        
        print("Converting to DataFrame...")
        df = convert_to_dataframe(benchmarks_data)
        
        # Save CSV locally
        csv_path = os.path.join('output', 'benchmarks_latest.csv')
        df.to_csv(csv_path, index=False)
        print(f"Saved CSV to {csv_path}")
        
        # Create and upload dataset
        print("Creating Hugging Face dataset...")
        dataset = create_hf_dataset(df)
        
        print(f"Uploading to {args.dataset}...")
        upload_to_hf(dataset, args.dataset, args.token, df)
        
        print("Upload complete!")
        print(f"Dataset available at: https://huggingface.co/datasets/{args.dataset}")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 