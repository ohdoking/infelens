# LLM Energy Consumption Measurement Tool

This project provides a comprehensive tool to measure and compare energy consumption, runtime performance, and CO2 emissions across different Language Learning Models (LLMs). It uses CodeCarbon for energy tracking and Hugging Face's Transformers library for model inference.

## Features

- Measure and compare energy consumption and CO2 emissions across multiple LLMs
- Support for various Hugging Face models with detailed model specifications
- Comprehensive benchmarking with diverse prompt categories
- Local testing mode with limited prompts for quick validation
- Detailed performance metrics and comparison summaries
- Poetry-based dependency management
- JSON output for detailed benchmark results
- Hardware-aware execution (CPU/GPU)
- Automatic upload to Hugging Face dataset for sharing and tracking

## Prerequisites

- Python 3.13 or higher
- Poetry (Python package manager)
- Git
- NVIDIA GPU (optional, for faster inference)
- Hugging Face account and API token

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd infelens
```

2. Install Poetry (if not already installed):
```bash
curl -sSL https://install.python-poetry.org | python3 -
```

3. Install project dependencies using Poetry:
```bash
poetry install
```

4. Activate the virtual environment:
```bash
poetry shell
```

5. Get your Hugging Face API token from https://huggingface.co/settings/tokens

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── poetry.lock
├── data/
│   ├── llm_models.json      # Model specifications and configurations
│   └── sample_prompts.json  # Diverse prompts for benchmarking
├── src/
│   ├── check_energy_consumption_script.py  # Core energy measurement script
│   ├── run_benchmarks.py                   # Multi-model benchmarking script
│   ├── upload_benchmarks_to_hf.py         # Upload script for Hugging Face
│   └── run_and_upload_benchmarks.py       # Combined benchmark and upload script
└── output/                                 # Generated benchmark results
```

## Usage

### Quick Start (Run Benchmarks and Upload)

To run benchmarks and upload results to Hugging Face in a single command:

```bash
poetry run python src/run_and_upload_benchmarks.py --token "your_hf_token"
```

Additional options:
```bash
# Run in local testing mode
poetry run python src/run_and_upload_benchmarks.py --local --token "your_hf_token"

# Run specific models
poetry run python src/run_and_upload_benchmarks.py --models "model1" "model2" --token "your_hf_token"

# Use custom dataset name
poetry run python src/run_and_upload_benchmarks.py --dataset "username/dataset-name" --token "your_hf_token"
```

### Running Benchmarks Only

To run benchmarks without uploading:

```bash
# Run all models
poetry run python src/run_benchmarks.py

# Run specific models
poetry run python src/run_benchmarks.py --models "model1" "model2"

# Run in local testing mode
poetry run python src/run_benchmarks.py --local

# Specify custom output file
poetry run python src/run_benchmarks.py --output "my_benchmarks.json"
```

### Uploading Existing Benchmarks

To upload previously generated benchmark results:

```bash
poetry run python src/upload_benchmarks_to_hf.py --token "your_hf_token"
```

The upload script will automatically use the latest benchmark results (`merged_benchmarks_latest.json`).

## Configuration

### Model Configuration

Models are configured in `data/llm_models.json`. Each model entry includes:
- Model name and Hugging Face identifier
- Parameter count
- Architecture details (hidden size, layers, etc.)
- Model type

Example model configuration:
```json
{
  "model_name": "TinyLlama-1.1B-Chat-v1.0",
  "huggingface_model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
  "num_params": "1.1B",
  "hidden_size": 2048,
  "num_layers": 24,
  "vocab_size": 32000,
  "seq_length": 2048,
  "model_type": "Transformer (TinyLlama)"
}
```

### Prompt Configuration

Prompts are stored in `data/sample_prompts.json` and are categorized by:
- Health and Wellness
- Education
- Science
- Environment
- Personal Development
- Technology
- Mathematics
- Creative Writing
- Programming

## Output

The benchmarking process generates:

1. Individual model metrics (per run):
   - Runtime statistics
   - Energy consumption
   - CO2 emissions
   - Per-prompt metrics
   - Hardware information

2. Merged benchmark results:
   - Timestamped version (e.g., `merged_benchmarks_20240220_123456.json`)
   - Latest version (`merged_benchmarks_latest.json`)
   - Both files contain the same data for easy reference

3. Hugging Face Dataset:
   - Automatically uploaded from latest benchmark results
   - Available at: https://huggingface.co/datasets/ohdoking/energy_consumption_by_model_and_gpu
   - Includes all metrics in a structured format
   - Updated with each new benchmark run

## Dependencies

Key dependencies managed by Poetry:
- transformers (^4.37.2)
- torch (^2.2.0)
- accelerate (^0.27.2)
- bitsandbytes (^0.42.0)
- codecarbon (^2.3.4)
- huggingface-hub (^0.32.3)
- datasets (^3.6.0)
- pandas (^2.2.3)

Development dependencies:
- pytest (^8.0.0)
- black (^24.1.1)
- isort (^5.13.2)
- flake8 (^7.0.0)

## Development

### Adding New Dependencies

```bash
poetry add package-name
```

### Updating Dependencies

```bash
poetry update
```

### Code Style

The project uses:
- Black for code formatting
- isort for import sorting
- flake8 for linting

Format code:
```bash
poetry run black src/
poetry run isort src/
```

## Notes

- The script automatically uses GPU if available, falling back to CPU if not
- Local testing mode uses 5 prompts for quick validation
- Energy consumption measurements are approximate and depend on hardware
- Results are timestamped for tracking performance over time
- Latest results are always available in both local files and Hugging Face dataset

## Troubleshooting

1. GPU-related issues:
   - Verify CUDA installation: `poetry run python -c "import torch; print(torch.cuda.is_available())"`
   - Check GPU memory: `nvidia-smi`
   - The script will automatically fall back to CPU if GPU is unavailable

2. Memory issues:
   - Use local testing mode (`--local`)
   - Try smaller models
   - Close other memory-intensive applications

3. CodeCarbon tracking issues:
   - Ensure proper system permissions
   - Verify power monitoring support
   - Try running with elevated privileges if needed

4. Hugging Face upload issues:
   - Verify your API token is correct
   - Check your internet connection
   - Ensure you have write access to the dataset repository

## License

(Add your license information here)

## Contributing

(Add contribution guidelines if applicable)
