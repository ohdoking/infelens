# LLM Energy Consumption Measurement Tool

This project provides a tool to measure energy consumption and runtime performance of Language Learning Models (LLMs) using a set of diverse prompts. It uses CodeCarbon for energy tracking and Hugging Face's Transformers library for model inference.

## Features

- Measure energy consumption and CO2 emissions of LLM inference
- Support for various Hugging Face models
- Local testing mode with limited prompts
- Detailed performance metrics
- Poetry-based dependency management
- CSV output for detailed energy consumption data

## Prerequisites

- Python 3.8 or higher
- Poetry (Python package manager)
- Git

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
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

## Project Structure

```
.
├── README.md
├── pyproject.toml
├── poetry.lock
├── sample_prompts.json
└── src/
    └── check_energy_consumption_script.py
```

## Usage

### Basic Usage

To run the script with default settings (TinyLlama model):

```bash
poetry run python src/check_energy_consumption_script.py
```

### Local Testing Mode

For local testing with limited prompts (5 prompts):

```bash
poetry run python src/check_energy_consumption_script.py --local
```

### Using a Different Model

To use a different model from Hugging Face:

```bash
poetry run python src/check_energy_consumption_script.py --model "model-name"
```

For local testing with a custom model:

```bash
poetry run python src/check_energy_consumption_script.py --local --model "model-name"
```

## Configuration

The script supports the following command-line arguments:

- `--local`: Run in local testing mode with 5 prompts
- `--model`: Specify the Hugging Face model to use (default: "TinyLlama/TinyLlama-1.1B-Chat-v1.0")

## Output

The script generates two types of output:

1. Console output showing:
   - Runtime statistics
   - Energy consumption
   - CO2 emissions
   - Per-prompt averages

2. CSV file (`emissions.csv`) containing detailed energy consumption data

## Dependencies

The project uses Poetry for dependency management. Key dependencies include:

- transformers
- torch
- accelerate
- bitsandbytes
- codecarbon

## Development

### Adding New Dependencies

To add a new dependency:

```bash
poetry add package-name
```

### Updating Dependencies

To update all dependencies:

```bash
poetry update
```

### Running Tests

(Add test instructions when tests are implemented)

## Notes

- The default model (TinyLlama-1.1B-Chat) is chosen for its small size and CPU compatibility
- For local testing, the script uses only 5 prompts to reduce resource usage
- Energy consumption measurements are approximate and depend on your hardware
- The script automatically uses CPU if no GPU is available

## Troubleshooting

1. If you encounter CUDA/GPU-related errors:
   - The script will automatically fall back to CPU
   - Check your PyTorch installation: `poetry run python -c "import torch; print(torch.cuda.is_available())"`

2. If you get memory errors:
   - Try running in local mode with `--local`
   - Use a smaller model
   - Close other memory-intensive applications

3. If CodeCarbon fails to track energy:
   - Ensure you have the necessary permissions
   - Check if your system supports power monitoring
   - Try running with administrator/sudo privileges

## License

(Add your license information here)

## Contributing

(Add contribution guidelines if applicable) # infelens
